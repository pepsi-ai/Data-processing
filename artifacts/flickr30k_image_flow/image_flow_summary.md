# Flickr30k Image Full Flow

- Images Used: 1000092795.jpg, 10002456.jpg, 1000268201.jpg, 1000344755.jpg, 1000366164.jpg
- Query Image: 1000092795.jpg
- Embedding Backend: open_clip
- Model: ViT-B-32 / laion2b_s34b_b79k
- CKKS Backend: tenseal
- Homomorphic Hash Backend: lthash_style_additive
- Ranking Homomorphic Consistent: True
- Ranking MAE: 0.0000017279

## Per-Image Homomorphic Hash Comparison

### 1000092795.jpg
- Plain Digest: 1e8ab0de60efa7de07e2d5459f3e19780e8a292e81a520a8199959df03db580a10cf7e1185b59c3e1f47314ccfddc20d04e41e05e7efc0a903eb2017cf778ab3
- CKKS Digest: 1e8ab0de60efa7de07e2d5459f3e19780e8a292e81a520a8199959df03db580a10cf7e1185b59c3e1f47314ccfddc20d04e41e05e7efc0a903eb2017cf778ab3
- Consistent: True
- Embedding Max Abs Diff: 0.000000005434
- Embedding Mean Abs Diff: 0.000000000856

### 10002456.jpg
- Plain Digest: 0120f87718746a381920bc691f4553ac0caffd06f58d2fd808f384998ce1e5210f78a74bd54c63c70df9c60efd89a5910a1df0a3599e7fe215aff222fe339392
- CKKS Digest: 0120f87718746a381920bc691f4553ac0caffd06f58d2fd808f384998ce1e5210f78a74bd54c63c70df9c60efd89a5910a1df0a3599e7fe215aff222fe339392
- Consistent: True
- Embedding Max Abs Diff: 0.000000005429
- Embedding Mean Abs Diff: 0.000000000911

### 1000268201.jpg
- Plain Digest: 04d26ad1aa2fecb104c1b1a6ab19b7a0136d81703feef72b119405f4174789641298ed5754fd016018fb4ffef08b58c712701fb5567a5cbf0b35abb74752c2f5
- CKKS Digest: 04d26ad1aa2fecb104c1b1a6ab19b7a0136d81703feef72b119405f4174789641298ed5754fd016018fb4ffef08b58c712701fb5567a5cbf0b35abb74752c2f5
- Consistent: True
- Embedding Max Abs Diff: 0.000000005717
- Embedding Mean Abs Diff: 0.000000000855

### 1000344755.jpg
- Plain Digest: 0cb38c6a4ade62151dfa2de2e3e8639503e905b1650d4df112f570003141e2c81e92f1bae509468a16bf49891914009414e3c6a9c60b3d570ea97323b212e3f1
- CKKS Digest: 0cb38c6a4ade62151dfa2de2e3e8639503e905b1650d4df112f570003141e2c81e92f1bae509468a16bf49891914009414e3c6a9c60b3d570ea97323b212e3f1
- Consistent: True
- Embedding Max Abs Diff: 0.000000006135
- Embedding Mean Abs Diff: 0.000000000904

### 1000366164.jpg
- Plain Digest: 0d5f8348b87c60ca11947cc71aeb6ae1134a5fae5073a2881312636225cfc35f19bb0b515265f43802ccb5b2615997190d1b1bdaa33e77e50e43ef499b3b82c7
- CKKS Digest: 0d5f8348b87c60ca11947cc71aeb6ae1134a5fae5073a2881312636225cfc35f19bb0b515265f43802ccb5b2615997190d1b1bdaa33e77e50e43ef499b3b82c7
- Consistent: True
- Embedding Max Abs Diff: 0.000000008984
- Embedding Mean Abs Diff: 0.000000000933

## Ranking Homomorphic Hash Comparison

- Plain Result Digest: 1bbdc46dcad92a6f15eaa75921c5b698122e1225ce5e11141ebcf9e1d9a73a36041f7bc00e5e6bc60a87be27317295c0109891ca26622a3a03620aa0e2441cd9
- CKKS Result Digest: 1bbdc46dcad92a6f15eaa75921c5b698122e1225ce5e11141ebcf9e1d9a73a36041f7bc00e5e6bc60a87be27317295c0109891ca26622a3a03620aa0e2441cd9
- Consistent: True
- Plain Ranked IDs: ['1000092795', '1000268201', '1000344755', '10002456', '1000366164']
- CKKS Ranked IDs: ['1000092795', '1000268201', '1000344755', '10002456', '1000366164']
