"""
Executor: Executes operations using LLM API.

在 MemSkill 的整体流程中，Controller 只负责选择 Top-K memory skills；
Executor 才是真正把这些 skills 落实为 MemoryBank 更新的模块。

本文件主要完成四件事：
1. 根据当前 span、检索到的历史 memory、Controller 选出的 skills 构造 LLM prompt。
2. 调用 LLM，让它输出 INSERT / UPDATE / DELETE / NOOP 等 memory actions。
3. 解析 LLM 返回的文本或 JSON，转换为结构化的 ExecutionResult。
4. 将这些结构化结果真正写回 MemoryBank。

注意：Executor 本身不是可训练神经网络，也不参与 PPO 反向传播；
它产生的执行结果会在 trainer 中进一步转化为 process reward，
间接影响 Controller 的 PPO 更新。
"""
import json
import re
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_repair import repair_json
from llm_utils import get_llm_response_via_api
from rag_utils import get_embeddings
from typing import List, Union


class ExecutionResult:
    """单条 memory action 的结构化执行结果。"""
    def __init__(self, action_type: str, success: bool,
                 memory_index: int = -1, memory_content: str = "",
                 reasoning: str = ""):
        # LLM 输出会先被解析成这个对象；此处只保存结果，不执行写库操作。
        self.action_type = action_type  # INSERT, UPDATE, DELETE, NOOP
        # success 表示解析或执行是否成功；失败 action 通常不会真正写入 MemoryBank。
        self.success = success
        # UPDATE / DELETE 使用 retrieved_memories 的局部下标。
        # apply_to_memory_bank 会通过 retrieved_indices 映射到 MemoryBank 的真实下标。
        self.memory_index = memory_index
        # INSERT 的新 memory 内容，或 UPDATE 后的完整 memory 内容。
        self.memory_content = memory_content
        # 失败原因或兼容旧格式的 LLM reasoning；当前 prompt 通常要求 LLM 不输出 reasoning。
        self.reasoning = reasoning

    def __repr__(self):
        # 调试打印时只截取 reasoning 前 100 个字符，避免日志过长。
        return f"ExecutionResult(action={self.action_type}, success={self.success}, " \
               f"mem_idx={self.memory_index}, reasoning={self.reasoning[:100]}...)"


class Executor:
    """Executor 负责调用 LLM，把 Controller 选出的 skills 执行为 memory actions。"""
    def __init__(self, args):
        # 保存命令行 / 配置参数，后续调用 LLM API 和计算 embedding 都会用到。
        self.args = args
        # 写入 MemoryBank 前，需要用同一个 retriever 为新 memory 计算检索向量。
        self.retriever_name = args.retriever
        self.logger = logging.getLogger('AgenticMemory')

    def _build_executor_prompt(self, operations: List, session_text: str,
                               retrieved_memories: List[str]) -> str:
        """根据 Controller 选出的 skills 构造 Executor prompt。"""
        # retrieved_memories 是 MemoryBank 检索出的相关记忆。
        # 这里给它们重新编号为 0, 1, 2...，LLM 后续只能引用这些局部编号。
        if len(retrieved_memories) > 0:
            mem_text = "\n".join([f"{i}. {mem}" for i, mem in enumerate(retrieved_memories)])
        else:
            mem_text = "(No existing memories retrieved)"

        skill_blocks = []
        seen = set()
        for op in operations:
            # operations 来自 Controller 选出的 Top-K skills。
            # 这里逐个读取 skill 的描述、指令模板和允许动作类型。
            name = getattr(op, "name", None)
            if not name:
                continue
            # 同名 skill 只保留一次，避免 prompt 中重复出现同一技能。
            if name in seen:
                continue
            seen.add(name)

            description = str(getattr(op, "description", "")).strip()
            instructions = str(getattr(op, "instruction_template", "")).strip()
            update_type = str(getattr(op, "update_type", "")).strip().upper()

            lines = [f"[Skill {len(seen)}] {name}"]
            if description:
                lines.append(f"Description: {description}")
            if update_type:
                lines.append(f"Allowed action: {update_type}")
            if instructions:
                lines.append("Instructions:")
                lines.append(instructions)
            skill_blocks.append("\n".join(lines))

        skills_text = "\n\n".join(skill_blocks) if skill_blocks else "(No skills provided)"

        # 这里要求 LLM 输出固定文本格式，后续 _parse_response 会按该格式解析。
        # 不让 LLM 输出解释，是为了降低解析难度并减少 memory action 中混入无关内容。
        return (
            "You are a memory management executor. Apply the selected skills to the input text\n"
            "chunk and retrieved memories, then output memory actions.\n\n"
            "Input Text Chunk:\n"
            f"{session_text}\n\n"
            "Retrieved Memories (0-based index):\n"
            f"{mem_text}\n\n"
            "Selected Skills:\n"
            f"{skills_text}\n\n"
            "Guidelines:\n"
            "- Apply any skill as needed; a skill may be used multiple times.\n"
            "- Read the input text chunk carefully line by line and apply any skill as needed.\n"
            "- Only use action types supported by the selected skills.\n"
            "- MEMORY_INDEX is 0-based and must reference the retrieved memories list.\n"
            "- Output only action blocks in the format below.\n"
            "- Do not include explanations or REASONING lines.\n"
            "Output format (repeat as needed). Use ONE block per action and separate blocks with"
            " a blank line:\n\n"
            "INSERT block:\n"
            "ACTION: INSERT\n"
            "MEMORY_ITEM: <concise but complete summary with essential details>\n\n"
            "UPDATE block:\n"
            "ACTION: UPDATE\n"
            "MEMORY_INDEX: <0-based index>\n"
            "UPDATED_MEMORY: <concise but complete merged summary with essential updates>\n\n"
            "DELETE block:\n"
            "ACTION: DELETE\n"
            "MEMORY_INDEX: <0-based index>\n\n"
        )

    def execute_operation(self, operation: Union[object, List[object]], session_text: str,
                          retrieved_memories: List[str]) -> List[ExecutionResult]:
        """调用 LLM 执行 Controller 选出的 memory skill，并返回结构化 actions。"""
        # Controller 可能只选出一个 skill，也可能选出 Top-K skills。
        # 这里统一整理成 list，后续 prompt 构造逻辑只处理 operations 列表。
        if isinstance(operation, (list, tuple)):
            operations = list(operation)
        else:
            operations = [operation]

        # 过滤空 skill。若没有任何有效 skill，则返回失败 NOOP，避免后续 LLM 调用无意义执行。
        operations = [op for op in operations if op is not None]
        if len(operations) == 0:
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="No operations provided"
            )]

        # 当前实现没有继续切分 session_text，因此 sub_chunks 只有当前 span / session 自身。
        sub_chunks = [session_text]
        all_results = []
        for sub_text in sub_chunks:
            if not sub_text.strip():
                continue
            # 把当前文本、retrieved memories、Controller 选出的 skills 合成 Executor prompt。
            instruction = self._build_executor_prompt(operations, sub_text, retrieved_memories)

            # 调用冻结的 LLM Executor。Executor 本身不训练参数，只通过 API 产生 memory actions。
            try:
                  response, _, _ = get_llm_response_via_api(
                      prompt=instruction,
                      LLM_MODEL=self.args.model,
                      base_url=self.args.api_base,
                      api_key=self.args.api_key,
                      MAX_TOKENS=self.args.max_new_tokens,
                      TAU=self.args.temperature,
                      MAX_TRIALS=10,
                      TIME_GAP=3,
                  )
            except Exception as e:
                # API 失败时返回 NOOP 失败结果，保证训练流程不会因为单次 LLM 调用中断。
                self.logger.warning(f"Executor API call failed: {e}")
                all_results.append(ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning=f"API call failed: {str(e)}"
                ))
                continue

            # LLM 可能一次返回多个 action block，这里统一解析为 ExecutionResult 列表。
            all_results.extend(self._parse_response(response, len(retrieved_memories)))

        if not all_results:
            # 如果空文本或解析失败导致没有任何结果，统一返回失败 NOOP，便于上层统计。
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="No executor results produced"
            )]
        return all_results

    def _parse_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
        """解析 LLM 返回的一个或多个 action block。"""
        response = self._normalize_response(response)
        results = []

        # Split response into individual action blocks.
        # Primary format: "ACTION: INSERT/UPDATE/DELETE/NOOP"
        action_pattern = re.compile(
            r'(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
            re.IGNORECASE
        )
        action_matches = list(action_pattern.finditer(response))

        # Compatibility format (no ACTION prefix):
        # INSERT
        # MEMORY_ITEM: ...
        if not action_matches:
            # 兼容没有 ACTION 前缀、只在单独一行写 INSERT/UPDATE/DELETE/NOOP 的旧格式。
            line_action_pattern = re.compile(
                r'(?im)^(?:[-*]\s*)?(INSERT|UPDATE|DELETE|NOOP)\s*(?::|=|-)?\s*$'
            )
            action_matches = list(line_action_pattern.finditer(response))

        if not action_matches:
            # 如果文本格式没有解析到 action，再尝试 JSON 格式。
            json_results = self._parse_json_response(response, num_retrieved)
            if json_results:
                return json_results
            self.logger.warning("ACTION: INSERT/UPDATE/DELETE/NOOP PARSE FAILED")
            print(response)
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="Failed to parse ACTION from response"
            )]

        # Parse each action block
        for i, match in enumerate(action_matches):
            # 当前 ACTION 到下一个 ACTION 之间的内容，就是一个 action block。
            block_start = match.start()
            block_end = action_matches[i + 1].start() if i + 1 < len(action_matches) else len(response)
            block = response[block_start:block_end].strip()

            # 单个 block 也可能解析出多条结果，例如一个 INSERT block 里列出多条 MEMORY_ITEM。
            result = self._parse_single_action(block, num_retrieved)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        return results

    def _normalize_response(self, response: str) -> str:
        """清理 LLM 输出的 Markdown 代码块包裹。"""
        text = response.replace("\r\n", "\n").strip()
        # 有些模型会返回 ```json ... ``` 或 ```text ... ```，
        # 解析前需要去掉最外层代码块。
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if "\n" in text:
                    first_line, rest = text.split("\n", 1)
                    if first_line.strip().lower() in ("json", "text"):
                        text = rest
        return text.strip()

    def _parse_json_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
        """兼容 JSON 格式的 LLM 输出。"""
        # 虽然 prompt 要求 action blocks，但实际 LLM 有时仍会返回 JSON。
        # 这里先从 response 中截取最外层 JSON 对象。
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return []
        try:
            json_str = response[json_start:json_end]
            # json_repair 用于修复轻微格式错误，例如多余逗号或缺少引号。
            repaired_json = repair_json(json_str)
            data = json.loads(repaired_json)
        except Exception:
            return []

        items = []
        if isinstance(data, dict):
            # 支持 {"actions": [...]}，也支持单个 {"action": ...}。
            if isinstance(data.get("actions"), list):
                items = data["actions"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return []

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", item.get("ACTION", ""))).strip().upper()
            if action == "INSERT":
                # INSERT 只需要新 memory 内容，不需要引用已有 memory 下标。
                content = str(item.get("memory_item", item.get("MEMORY_ITEM", ""))).strip()
                if not content:
                    continue
                results.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=content,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "UPDATE":
                # UPDATE 必须同时包含 retrieved memory 的局部下标和更新后的完整内容。
                idx = item.get("memory_index", item.get("MEMORY_INDEX", None))
                content = str(item.get("updated_memory", item.get("UPDATED_MEMORY", ""))).strip()
                if idx is None or content == "":
                    continue
                try:
                    idx = int(idx)
                except Exception:
                    continue
                # idx 是 retrieved_memories 的局部下标，范围必须落在 [0, num_retrieved)。
                if idx < 0 or idx >= num_retrieved:
                    continue
                results.append(ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=idx,
                    memory_content=content,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "DELETE":
                # DELETE 只需要 retrieved memory 的局部下标，但必须保证下标没有越界。
                idx = item.get("memory_index", item.get("MEMORY_INDEX", None))
                if idx is None:
                    continue
                try:
                    idx = int(idx)
                except Exception:
                    continue
                # idx 仍然是 retrieved_memories 的局部下标。
                if idx < 0 or idx >= num_retrieved:
                    continue
                results.append(ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=idx,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "NOOP":
                results.append(ExecutionResult(
                    action_type="NOOP",
                    success=True,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))

        return results

    def _parse_single_action(self, block: str, num_retrieved: int) -> ExecutionResult:
        """解析单个 action block。"""
        # 优先解析显式格式：ACTION: INSERT / UPDATE / DELETE / NOOP。
        action_match = re.search(
            r'ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
            block,
            re.IGNORECASE
        )

        # Backward-compatible fallback: line-only action marker.
        if not action_match:
            # 兼容旧格式：单独一行 INSERT / UPDATE / DELETE / NOOP。
            action_match = re.search(
                r'(?im)^(?:[-*]\s*)?(INSERT|UPDATE|DELETE|NOOP)\s*(?::|=|-)?\s*$',
                block
            )

        if not action_match:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="Failed to parse ACTION from block"
            )

        action_type = action_match.group(1).upper()

        # Extract REASONING
        # 当前 prompt 要求不输出 REASONING，但这里保留兼容逻辑，方便解析旧响应和调试失败原因。
        reasoning_match = re.search(
            r'REASONING\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*(?:ACTION\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|'
            r'UPDATED[_ ]MEMORY\s*(?::|=|-)?|MEMORY[_ ]INDEX\s*(?::|=|-)?|$))',
            block,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        # Parse based on action type
        if action_type == "NOOP":
            # NOOP 表示当前 span 不需要修改 MemoryBank。成功 NOOP 是一种合法执行结果。
            return ExecutionResult(
                action_type="NOOP",
                success=True,
                reasoning=reasoning
            )

        elif action_type == "INSERT":
            # INSERT 允许一个 block 中包含多个 MEMORY_ITEM，因此这里可能返回多条结果。
            # 支持 MEMORY_ITEM / NEW_MEMORY / CONTENT / MEMORY 等字段名，是为了兼容 LLM 输出波动。
            memory_pattern = re.compile(
                r'(?:MEMORY[_ ]ITEM|NEW[_ ]MEMORY|CONTENT|MEMORY)\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*'
                r'(?:REASONING\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|UPDATED[_ ]MEMORY\s*(?::|=|-)?|'
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?|ACTION\s*(?::|=|-)?|$))',
                re.IGNORECASE | re.DOTALL
            )
            memory_matches = [m.strip() for m in memory_pattern.findall(block) if m.strip()]
            if not memory_matches:
                # 兼容模型没有严格写 MEMORY_ITEM 标签、而是直接列出内容的情况。
                fallback_lines = []
                for line in block.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if re.match(r'^(ACTION|MEMORY[_ ]ITEM|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX|REASONING)\b', stripped, re.IGNORECASE):
                        continue
                    if re.match(r'^\w+\s+block\s*:?\s*$', stripped, re.IGNORECASE):
                        continue
                    stripped = re.sub(r'^[-*•]\s*', '', stripped)
                    stripped = re.sub(r'^\d+\.\s*', '', stripped)
                    if stripped:
                        fallback_lines.append(stripped)
                memory_matches = fallback_lines
            if not memory_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_ITEM for INSERT"
                )

            if len(memory_matches) == 1:
                return ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=memory_matches[0],
                    reasoning=reasoning
                )

            results = []
            for mem in memory_matches:
                results.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=mem,
                    reasoning=reasoning
                ))
            return results

        elif action_type == "UPDATE":
            # UPDATE 需要把 retrieved memory 的局部编号和新的完整 memory 内容成对解析。
            index_matches = re.findall(
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)',
                block,
                re.IGNORECASE
            )
            update_pattern = re.compile(
                r'UPDATED[_ ]MEMORY\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*(?:REASONING\s*(?::|=|-)?|'
                r'UPDATED[_ ]MEMORY\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|'
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?|ACTION\s*(?::|=|-)?|$))',
                re.IGNORECASE | re.DOTALL
            )
            update_matches = [m.strip() for m in update_pattern.findall(block) if m.strip()]

            if not index_matches or not update_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_INDEX/UPDATED_MEMORY for UPDATE"
                )

            pair_count = min(len(index_matches), len(update_matches))
            results = []
            for i in range(pair_count):
                memory_index = int(index_matches[i])
                if memory_index >= num_retrieved:
                    # 这里校验的是 retrieved_memories 的局部下标，不是 MemoryBank 全局下标。
                    results.append(ExecutionResult(
                        action_type="NOOP",
                        success=False,
                        reasoning=f"MEMORY_INDEX {memory_index} out of range [0, {num_retrieved})"
                    ))
                    continue
                results.append(ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=memory_index,
                    memory_content=update_matches[i],
                    reasoning=reasoning
                ))

            if len(results) == 1:
                return results[0]
            return results

        elif action_type == "DELETE":
            # DELETE 可以一次删除多个 retrieved memories，因此也可能返回多条结果。
            index_matches = re.findall(
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)',
                block,
                re.IGNORECASE
            )
            if not index_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_INDEX for DELETE"
                )

            results = []
            for idx_str in index_matches:
                memory_index = int(idx_str)
                if memory_index >= num_retrieved:
                    results.append(ExecutionResult(
                        action_type="NOOP",
                        success=False,
                        reasoning=f"MEMORY_INDEX {memory_index} out of range [0, {num_retrieved})"
                    ))
                    continue
                results.append(ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=memory_index,
                    reasoning=reasoning
                ))

            if len(results) == 1:
                return results[0]
            return results

        else:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning=f"Unknown action type: {action_type}"
            )

    def apply_to_memory_bank(self, results: List[ExecutionResult],
                              memory_bank, retrieved_indices: List[int],
                              operation_name: Union[str, List[str], None] = None) -> bool:
        """将结构化 ExecutionResult 真正应用到 MemoryBank。"""
        if not results:
            return True

        # 按 action 类型分组。写库顺序不能完全按 LLM 输出顺序来：
        # UPDATE 要先做，DELETE 要倒序做，INSERT 最后做。
        inserts = [r for r in results if r.action_type == "INSERT" and r.success]
        updates = [r for r in results if r.action_type == "UPDATE" and r.success]
        deletes = [r for r in results if r.action_type == "DELETE" and r.success]
        # print(len(inserts), len(updates), len(deletes))

        all_success = True

        # Batch compute embeddings for UPDATE and INSERT operations
        # Collect all contents that need embeddings
        # UPDATE 和 INSERT 都会产生新的 memory 内容，因此都需要重新计算 embedding。
        update_contents = [r.memory_content for r in updates]
        insert_contents = [r.memory_content for r in inserts]
        all_contents = update_contents + insert_contents

        # Batch compute retriever embeddings
        all_retriever_embeddings = []
        if all_contents:
            try:
                all_retriever_embeddings = get_embeddings(
                    self.retriever_name,
                    all_contents,
                    'context'
                )
            except Exception as e:
                self.logger.warning(f"Failed to batch compute retriever embeddings: {e}")
                all_success = False
                # Fall back to empty embeddings (operations will fail individually)
                all_retriever_embeddings = [None] * len(all_contents)

        # Batch compute state encoder embeddings if encoder is available
        # state_encoder embedding 是 Controller 构造 state 时会用到的向量缓存；
        # 如果这里能提前算好，后续 MemoryBank 检索和状态构造会更直接。
        all_state_encoder_embeddings = []
        if all_contents and memory_bank.state_encoder is not None:
            try:
                all_state_encoder_embeddings = memory_bank.state_encoder._encode_texts(all_contents)
            except Exception as e:
                self.logger.warning(f"Failed to batch compute state encoder embeddings: {e}")
                # Fall back to None (memory_bank will compute individually if needed)
                all_state_encoder_embeddings = [None] * len(all_contents)
        else:
            all_state_encoder_embeddings = [None] * len(all_contents)

        # Split embeddings back to update and insert
        # 前面为了减少调用次数把 UPDATE 和 INSERT 内容合并成一个 batch；
        # 这里再按原顺序切回各自对应的 embedding。
        update_retriever_embeddings = all_retriever_embeddings[:len(updates)]
        update_state_encoder_embeddings = (
            all_state_encoder_embeddings[:len(updates)] if len(all_state_encoder_embeddings) > 0 else [None] * len(updates)
        )
        insert_retriever_embeddings = all_retriever_embeddings[len(updates):]
        insert_state_encoder_embeddings = (
            all_state_encoder_embeddings[len(updates):] if len(all_state_encoder_embeddings) > 0 else [None] * len(inserts)
        )

        # 1. Process UPDATEs first (before any DELETEs change indices)
        for i, result in enumerate(updates):
            try:
                # result.memory_index 是 retrieved_memories 中的局部下标；
                # actual_index 才是 MemoryBank 内部真实下标。
                actual_index = retrieved_indices[result.memory_index]
                retriever_emb = update_retriever_embeddings[i]
                state_encoder_emb = (
                    update_state_encoder_embeddings[i] if i < len(update_state_encoder_embeddings) else None
                )
                if retriever_emb is None:
                    raise ValueError("Retriever embedding is None")
                memory_bank.update_memory(
                    index=actual_index,
                    new_content=result.memory_content,
                    new_embedding=retriever_emb,
                    new_state_encoder_embedding=state_encoder_emb,
                    operation_name=operation_name
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply UPDATE: {e}")
                all_success = False

        # 2. Process DELETEs in reverse order of *actual* memory bank indices to avoid index shift issues
        delete_targets = []
        for result in deletes:
            try:
                # 同样先把 LLM 输出的局部下标映射为 MemoryBank 全局下标。
                actual_index = retrieved_indices[result.memory_index]
                delete_targets.append((actual_index, result))
            except Exception as e:
                self.logger.warning(f"Failed to map DELETE target: {e}")
                all_success = False

        for actual_index, result in sorted(delete_targets, key=lambda x: x[0], reverse=True):
            try:
                memory_bank.delete_memory(index=actual_index)
            except Exception as e:
                self.logger.warning(f"Failed to apply DELETE: {e}")
                all_success = False

        # 3. Process INSERTs last (doesn't affect existing indices)
        for i, result in enumerate(inserts):
            try:
                # INSERT 是新增 memory，不需要 retrieved_indices 映射。
                retriever_emb = insert_retriever_embeddings[i]
                state_encoder_emb = (
                    insert_state_encoder_embeddings[i] if i < len(insert_state_encoder_embeddings) else None
                )
                if retriever_emb is None:
                    raise ValueError("Retriever embedding is None")
                memory_bank.add_memory(
                    content=result.memory_content,
                    embedding=retriever_emb,
                    state_encoder_embedding=state_encoder_emb,
                    metadata={'source': 'inserted'},
                    operation_name=operation_name
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply INSERT: {e}")
                all_success = False

        return all_success
