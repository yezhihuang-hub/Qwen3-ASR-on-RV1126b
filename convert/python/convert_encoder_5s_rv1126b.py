from rknn.api import RKNN

onnx_model = "build_rv1126b/qwen3_asr_encoder_merged.fp16.5s.rv1126b.onnx"
out_model = "build_rv1126b/qwen3_asr_encoder_merged.fp16.5s.rv1126b.rknn"

rknn = RKNN(verbose=True)

print("[1/4] config")
ret = rknn.config(
    target_platform="rv1126b",
    optimization_level=3,
)
if ret != 0:
    print("config failed", ret)
    exit(ret)

print("[2/4] load onnx")
ret = rknn.load_onnx(
    model=onnx_model,
    inputs=["input_features"],
    input_size_list=[[1, 128, 500]],
)
if ret != 0:
    print("load_onnx failed", ret)
    exit(ret)

print("[3/4] build")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print("build failed", ret)
    exit(ret)

print("[4/4] export")
ret = rknn.export_rknn(out_model)
if ret != 0:
    print("export failed", ret)
    exit(ret)

rknn.release()
print("DONE:", out_model)
