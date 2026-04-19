# `项目方案.md` 生成计划：V1 聚焦数据处理与同态加密

## 摘要
- 输出文件固定为 `E:\数据处理2\项目方案.md`，中文，定位为“研究+工程并重”的本地原型方案。
- `V1` 目标收敛为“数据处理闭环”，只做本地图文双模态的规范化、特征提取、鲁棒哈希、同态加密与加密相似度实验。
- 上链、存证、验证、链下对象管理全部移到后续阶段，在 `V1` 文档中仅保留扩展接口，不进入首期实施与验收。
- 文档中明确写出：严格哈希不能解决 JPEG/PNG 这类有损跨格式一致性，跨格式能力要依赖共享嵌入与鲁棒哈希。

## V1 方案内容
- `1. 项目背景与研究问题`
  - 聚焦“多模态数据在本地环境下经过统一处理后，能否支持稳定表征、鲁棒检索和密文相似度计算”。
- `2. 可行性分析`
  - 可行：图文共享嵌入、本地 CKKS 向量加密、明密文相似度近似一致。
  - 部分可行：跨格式严格一致性不可承诺，只能做鲁棒一致性。
  - 暂不做：链上登记、链上验证、联盟链部署。
- `3. 本地数据处理架构`
  - `原始数据 -> 规范化 -> 嵌入生成 -> 双哈希 -> 向量加密 -> 加密相似度实验 -> 结果评估`
- `4. 核心技术路线`
  - 图像：解码、去元数据、转 `RGB`、缩放 `224x224`
  - 文本：`Unicode NFC`、空白折叠、统一编码
  - 嵌入：`OpenCLIP ViT-B/32`
  - 哈希：`file_sha256`、`canonical_sha256`、`semantic_hash_128`
  - 加密：`CKKS` 本地加密与密文相似度计算
- `5. 实验设计`
  - 只围绕数据处理与加密阶段设计对照组和指标
- `6. 后续扩展接口`
  - 为后续上链和验证预留数据结构，但不实现

## 关键模块与接口
- `Canonicalizer.normalize(input_path, modality) -> canonical_path, meta`
- `Embedder.encode(canonical_path, modality) -> embedding_256`
- `Hasher.build(raw_path, canonical_path, embedding) -> hash_record`
- `Encryptor.encrypt(embedding_256) -> ciphertext`
- `Encryptor.similarity(cipher_a, cipher_b) -> encrypted_score_or_decrypted_score`
- `ExperimentRunner.run(config) -> metrics_report`
- 预留但不实现：
  - `LedgerAdapter.register(hash_record)`
  - `Verifier.verify(asset_id)`

## 实验与验收
- 数据集固定为本地图文双模态：
  - 主集：`MS COCO`
  - 图像变体：`PNG/JPEG(95/75/50)/WebP`
  - 文本变体：大小写、空白、标点扰动
- 对照实验固定 4 组：
  - `Exp1` 原始文件 `SHA-256`
  - `Exp2` 规范化后 `canonical_sha256`
  - `Exp3` `pHash`
  - `Exp4` `OpenCLIP` 嵌入 + `semantic_hash_128` + `CKKS`
- 核心指标：
  - 规范化一致率
  - 跨格式检索 `Recall@K`
  - 明文与密文相似度误差
  - 加密前后排序一致性
  - 本地单次编码、加密、检索耗时
- 首期验收阈值：
  - 无损重编码下 `canonical_sha256` 一致率 `100%`
  - 跨格式检索 `Recall@10 >= 0.90`
  - 明密文相似度误差 `<= 0.03`
  - 本地完整处理链路可重复跑通

## 实施排期与默认假设
- 周期固定 4 周：
  - 第 1 周：数据准备与规范化
  - 第 2 周：基线哈希、`pHash`、共享嵌入
  - 第 3 周：`semantic_hash_128` 与 `CKKS` 加密相似度
  - 第 4 周：实验复现、指标汇总、方案文档定稿
- 默认技术栈固定为 `Python + PyTorch + OpenCLIP + OpenFHE + 本地文件存储`
- 默认不做数据库、区块链、服务化部署；若需要持久化，只保留本地结果文件与实验报告
- 后续阶段再增加 `SQLite/MinIO/Hyperledger Fabric`，但这些不应写入 `V1` 的实施承诺
