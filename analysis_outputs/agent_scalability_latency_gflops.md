# Agent Scalability (2026-04-21T07:21:25)

|method|agent_n|status|latency_ms|gflops|error|
|---|---:|---|---:|---:|---|
|HAM|2|OK|6.2874|NA||
|HAM|4|OK|6.1031|NA||
|HAM|6|OK|6.1616|NA||
|HAM|8|OK|6.3021|NA||
|AttFusion|2|OK|0.2776|NA||
|AttFusion|4|OK|0.2903|NA||
|AttFusion|6|OK|0.3841|NA||
|AttFusion|8|OK|0.4398|NA||
|DiscoNet|2|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.disco_fuse'|
|DiscoNet|4|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.disco_fuse'|
|DiscoNet|6|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.disco_fuse'|
|DiscoNet|8|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.disco_fuse'|
|CoBEVT|2|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.swap_fusion_modules'|
|CoBEVT|4|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.swap_fusion_modules'|
|CoBEVT|6|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.swap_fusion_modules'|
|CoBEVT|8|FAIL|NA|NA|build_fail:No module named 'opencood.models.fuse_modules.swap_fusion_modules'|
|V2XViT|2|FAIL|NA|NA|build_fail:No module named 'opencood.models.sub_modules.v2xvit_basic'|
|V2XViT|4|FAIL|NA|NA|build_fail:No module named 'opencood.models.sub_modules.v2xvit_basic'|
|V2XViT|6|FAIL|NA|NA|build_fail:No module named 'opencood.models.sub_modules.v2xvit_basic'|
|V2XViT|8|FAIL|NA|NA|build_fail:No module named 'opencood.models.sub_modules.v2xvit_basic'|
