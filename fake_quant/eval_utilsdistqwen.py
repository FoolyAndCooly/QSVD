# File: eval_utilsdistqwen.py
import torch
import os
import logging
import json
from tqdm import tqdm
from vlmeval.smp import dump, get_rank_and_world_size, string, load
import torch.distributed as dist
from PIL import Image
import pandas as pd

# @torch.no_grad() 装饰器表示该函数内的所有 PyTorch 操作都不会计算梯度，以节省显存并加速。
@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    """
    Qwen2.5-VL 模型评估器。

    该函数负责在多选问答数据集上对 Qwen-VL 模型进行分布式推理和评估。

    Args:
        model: 要评估的 Qwen2.5-VL 模型对象。
        testenc: 封装了评测数据集和评测逻辑的对象 (e.g., from VLMEvalKit)。
        dev: 计算设备 (未使用，因为 rank 会被用于设备选择)。
        args: 包含各种配置参数的对象 (e.g., save_path, case_study)。
        tokenizer: 此处应为 Qwen-VL 的统一 processor 对象。
        image_processor: 此处也应为 Qwen-VL 的统一 processor 对象 (与 tokenizer 是同一个对象)。
    """
    # 为了代码清晰，我们将 processor 明确命名。在 Qwen-VL 中，tokenizer 和 image_processor 是同一个对象。
    processor = image_processor

    # 获取当前进程在分布式环境中的排名 (rank) 和总进程数 (world_size)。
    rank, world_size = get_rank_and_world_size()
    # testenc 是一个元组 (dataloader, dataset_object)，这里解包获取数据集对象。
    _, testenc = testenc
    # 将模型移动到当前进程对应的 GPU 上。
    print(f"rank: {rank}")
    model = model.to(torch.device(f'cuda:{rank}'))
    logging.info(f"Model moved to rank {rank}.")

    # 1. 辅助函数：构建符合 Qwen-VL 聊天模板的消息列表
    def build_prompt_qwen(line, dataset):
        """
        为多选问答任务构建一个标准的消息列表 (list of dictionaries)。
        这个格式是通用的，可以被 Qwen-VL 的 processor.apply_chat_template 使用。
        """
        # 从数据集中获取并保存图像的临时路径。
        tgt_path = dataset.dump_image(line)
        # 提取问题文本。
        question = line['question']
        # 检查是否有提示（hint），如果有，就加在问题前面。
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question
        
        # 提取所有有效的选项 (A, B, C, ...)。
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        # 将选项格式化并追加到问题后面。
        for key, item in options.items():
            question += f"\n{key}. {item}"
        
        # 如果不是进行案例研究，就追加一个指令，让模型直接回答选项字母。
        if not args.case_study:
            question += "\nAnswer with the option's letter from the given choices directly."
        
        # 构建一个消息列表。首先是图像部分。
        message = [dict(type="image", value=s) for s in tgt_path]
        # 然后是文本部分。
        message.append(dict(type="text", value=question))
        return message

    # 2. 辅助函数：清理模型输出
    def output_process_qwen(answer):
        """
        清理 Qwen-VL 模型生成的原始文本，移除特殊的模板和结束标记。
        """
        # Qwen-VL 的聊天模板在 assistant 回答前没有特殊前缀，但会以 <|im_end|> 或 </s> 结束。
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        return answer.strip().rstrip('.')

    # 3. 辅助函数：将消息列表转换为模型输入
    def message_to_prompt_qwen(data, processor):
        """
        使用 Qwen-VL 的统一 processor 将消息列表和图像转换为模型所需的输入张量。
        """
        content, images = [], []
        # 遍历消息列表，分离文本和图像。
        for msg in data:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image", "image": Image.open(msg["value"]).convert("RGB")}) # 为聊天模板准备图像占位符
                images.append(Image.open(msg["value"]).convert("RGB"))
        
        # 构建符合 Qwen 聊天模板的对话结构。
        conversation = [{"role": "user", "content": content}]
        
        # 使用 processor 的 `apply_chat_template` 方法生成包含特殊 token 的文本 prompt。
        # add_generation_prompt=True 会在末尾添加 '<|im_start|>assistant\n'，引导模型生成回答。
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        
        # 将文本 prompt 和 PIL 图像列表一起传递给 processor。
        # 它会自动完成分词、图像预处理，并将它们整合成模型所需的 `input_ids`, `attention_mask`, `pixel_values`。
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
        return inputs

    # 4. 主推理函数
    def infer_data_qwen(model, processor, args, verbose=False):
        """
        执行分布式推理的核心循环。
        """
        dataset = testenc
        rank, world_size = get_rank_and_world_size()

        if rank == 0:
            logging.info("Start selecting split data for each rank.")
        
        # --- 数据集切分：每个 GPU 进程只获取数据集的一个子集 ---
        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]

        # --- 结果文件和断点续传处理 ---
        if args.case_study:
            case_file = os.path.join(args.save_path, 'case_study.jsonl')
            os.makedirs(os.path.dirname(case_file), exist_ok=True)
            case_anw = open(case_file, "w")
            res = {}
        else:
            out_file = os.path.join(args.save_path, f'vlm_eval_rank{rank}.pkl')
            data_indices = [i for i in data['index']]
            res = {}
            if os.path.exists(out_file):
                res.update(load(out_file)) # 加载已有结果，实现断点续传
            data = data[~data['index'].isin(res)] # 从待处理数据中移除已完成的

        if world_size > 1:
            dist.barrier() # 确保所有进程都完成了数据准备

        if rank == 0:
            logging.info("Finished data selection. Starting inference.")
        
        # --- 推理循环 ---
        for i in tqdm(range(len(data)), disable=(rank != 0)):
            # 获取数据索引和内容
            idx = data.iloc[i]['index']
            # 构建 prompt 消息列表
            message = build_prompt_qwen(data.iloc[i], dataset)
            
            # 将消息列表转换为模型输入
            inputs = message_to_prompt_qwen(message, processor)
            input_token_len = inputs['input_ids'].shape[1]
            # 调用模型的 generate 方法进行推理
            output = model.generate(
                **inputs,
                do_sample=False, 
                temperature=0,
                max_new_tokens=16 if not args.case_study else 512, # 多选题答案短，限制长度；案例研究则需要更长输出
                num_beams=1
            )
            
            # 解码模型输出的 token IDs 为文本
            newly_generated_ids = output[0, input_token_len:]
            response = processor.decode(newly_generated_ids, skip_special_tokens=True)
            # 清理和后处理文本
            response = output_process_qwen(response)

            # --- 保存结果 ---
            if args.case_study:
                case_anw.write(json.dumps({"case_id": i, "input": message, "output": response}) + "\n")
            else:
                res[idx] = response
                if (i + 1) % 10 == 0: # 定期保存，防止意外中断
                    dump(res, out_file)

        # --- 推理结束后的收尾工作 ---
        if args.case_study:
            logging.info("Case study finished.")
            case_anw.close()
            return
        
        dump({k: res[k] for k in data_indices}, out_file) # 保存最后剩余的结果
        if world_size > 1:
            dist.barrier() # 等待所有进程完成推理

        # --- 结果合并与清理 (仅在 rank 0 进程执行) ---
        if rank == 0:
            merged_results = {}
            for r in range(world_size):
                rank_file = os.path.join(args.save_path, f'vlm_eval_rank{r}.pkl')
                if os.path.exists(rank_file):
                    merged_results.update(load(rank_file))
                    os.remove(rank_file) # 合并后删除临时文件
            
            merged_out_file = os.path.join(args.save_path, 'vlm_eval.pkl')
            dump(merged_results, merged_out_file)
            logging.info("Merged and cleaned up all per-GPU outputs.")
        return

    # ==========================================================
    #                  主函数 `evaluator` 执行流程
    # ==========================================================

    if world_size > 1:
        dist.barrier() # 开始前同步所有进程
    
    # 执行推理
    infer_data_qwen(model, processor, args, verbose=True)
    
    if args.case_study:
        return # 案例研究模式下提前退出

    if world_size > 1:
        dist.barrier() # 推理结束后同步

    # --- 最终评估 (仅在 rank 0 进程执行) ---
    if rank == 0:
        logging.info("Starting final evaluation on rank 0.")
        result_file = os.path.join(args.save_path, 'vlm_result.xlsx')
        data_all = load(os.path.join(args.save_path, 'vlm_eval.pkl'))
        dataset_df = testenc.data

        # 将预测结果添加到数据集的 DataFrame 中
        dataset_df['prediction'] = [str(data_all[x]) for x in dataset_df['index']]
        
        # 保存为 Excel 文件，方便人工查阅
        dump(dataset_df, result_file)

        # 调用数据集对象的 evaluate 方法进行自动评估
        judge_kwargs = {'nproc': 4, 'verbose': True}
        eval_results = testenc.evaluate(result_file, **judge_kwargs)
        
        # 打印格式化的评估结果
        logging.info("Evaluation finished. Results:")
        if isinstance(eval_results, dict):
            logging.info('\n' + json.dumps(eval_results, indent=4))
        return eval_results
