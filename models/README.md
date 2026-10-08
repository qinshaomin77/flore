# 模型

grid36 为固定场景目标/消融模型；grid36_demand 为需求匹配模型；kunshan_spatial 为昆山 w100/w300/w500 模型。

使用对应 configs 配置。文件名中 full_labmda_0p8 保留原始拼写，以保持权重身份。训练新模型优先选训练输出的 checkpoint_best.pt，选择标准记录在 run_config 与 best 日志中。

PyTorch checkpoint 应只从可信发布来源加载。SHA256 见 data/manifest.json。不要将旧昆山模型交给 Grid36 实现加载。

旧敏感性/多种子权重缺少原阈值，未纳入；configs/experiments.yaml 提供这些实验的重训与评估链路。
