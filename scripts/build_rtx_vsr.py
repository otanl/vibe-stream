"""Build out/rtx_vsr/rtxvsr.dll: NVIDIA RTX Video Super Resolution callable from Python (vibe_stream.upscale.RTXVSR).

The RTX Video SDK (https://developer.nvidia.com/rtx-video-sdk) is not part of this repository; its licence allows
modifying the SDK's samples and shipping the result in object code inside an application. This script copies the
SDK's CUDA sample (samples/RTX_Video_API), patches it into a DLL body (errors come back as a failed call instead of
waiting for a key and ending the process; NGX looks for nvngx_vsr.dll next to the DLL), builds it with CMake (Visual
Studio 2022, CUDA 12) and copies the SDK's nvngx_vsr.dll beside it. Running it needs an RTX GPU and, for the CUDA
path, driver R570 or later.

Usage: python scripts/build_rtx_vsr.py --sdk C:/path/to/RTX_Video_SDK_v1.1.0 [--out out/rtx_vsr]
"""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import _paths  # noqa: F401
from vibe_stream.vibe_model import ROOT

SAMPLE = ("rtx_video_api_cuda_impl.cpp", "rtx_video_api.h", "utils.h")
EXPORTS = """LIBRARY rtxvsr
EXPORTS
  rtx_video_api_cuda_create
  rtx_video_api_cuda_evaluate
  rtx_video_api_cuda_evaluate_deviceptr
  rtx_video_api_cuda_shutdown
"""
CMAKE = """cmake_minimum_required(VERSION 3.20)
project(rtxvsr CXX)
find_package(CUDAToolkit REQUIRED)
add_library(rtxvsr SHARED src/rtx_video_api_cuda_impl.cpp src/rtxvsr.def)
target_include_directories(rtxvsr PRIVATE "{sdk}/include" src)
target_link_libraries(rtxvsr PRIVATE CUDA::cuda_driver CUDA::cudart_static "{sdk}/lib/Windows/x64/nvsdk_ngx_d.lib")
"""
# NGX_Init with the folder of this DLL as the feature search path (NVSDK_NGX_FeatureCommonInfo)
INIT = r'''    static std::wstring dir;
    if (dir.empty())
    {
        HMODULE h = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&rtx_video_api_cuda_create, &h);
        wchar_t buf[MAX_PATH];
        GetModuleFileNameW(h, buf, MAX_PATH);
        dir = buf;
        dir = dir.substr(0, dir.find_last_of(L"\\/"));
    }
    static const wchar_t* paths[1];
    paths[0] = dir.c_str();
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    CHECK_NGX(NVSDK_NGX_CUDA_Init(APP_ID, dir.c_str(), &info));'''


def patch(s: str) -> str:
    s = s.replace("#include <cstring>\n", "#include <cstring>\n#include <string>\n#include <stdexcept>\n#include <windows.h>\n", 1)
    s = re.sub(r"getchar\(\);\s*\\\s*\n\s*exit\(EXIT_FAILURE\);", 'throw std::runtime_error("ngx");', s)
    s = re.sub(r"exit\(0\);", 'throw std::runtime_error("cuda");', s)
    init = "    CHECK_NGX(NVSDK_NGX_CUDA_Init(APP_ID, APP_PATH));"
    if init not in s:
        raise SystemExit("the sample's NGX init line was not found: another SDK version?")
    s = s.replace(init, INIT)
    for call in ("create(cuContext, cuStream, iGpu, THDREnable, VSREnable)",
                 "evaluate_deviceptr(cuDeviceptr_Input, cuDeviceptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting)",
                 "evaluate(cuTexObject_Input, cuSurfObject_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting)",
                 "evaluate_hostptr(hostptr_Input, hostptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting)"):
        line = f"return p_cuda_api_impl->{call};"
        if line not in s:
            raise SystemExit(f"not found in the sample: {line}")
        s = s.replace(line, "try { " + line + " } catch (...) { return API_BOOL_FAIL; }")
    return s.replace("        p_cuda_api_impl->shutdown();", "        try { p_cuda_api_impl->shutdown(); } catch (...) {}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", default=os.environ.get("NV_RTX_VIDEO_SDK"), help="the unpacked RTX Video SDK")
    ap.add_argument("--out", default=str(ROOT / "out" / "rtx_vsr"))
    a = ap.parse_args()
    if not a.sdk:
        ap.error("give --sdk (or set NV_RTX_VIDEO_SDK)")
    sdk, out = Path(a.sdk), Path(a.out)
    (out / "src").mkdir(parents=True, exist_ok=True)
    for f in SAMPLE:
        shutil.copy(sdk / "samples" / "RTX_Video_API" / f, out / "src" / f)
    p = out / "src" / SAMPLE[0]
    p.write_text(patch(p.read_text(encoding="utf-8")), encoding="utf-8")
    (out / "src" / "rtxvsr.def").write_text(EXPORTS, encoding="utf-8")
    (out / "CMakeLists.txt").write_text(CMAKE.format(sdk=sdk.as_posix()), encoding="utf-8")
    subprocess.run(["cmake", "-S", str(out), "-B", str(out / "build"), "-G", "Visual Studio 17 2022", "-A", "x64"],
                   check=True)
    subprocess.run(["cmake", "--build", str(out / "build"), "--config", "Release"], check=True)
    shutil.copy(out / "build" / "Release" / "rtxvsr.dll", out / "rtxvsr.dll")
    shutil.copy(sdk / "bin" / "Windows" / "x64" / "rel" / "nvngx_vsr.dll", out / "nvngx_vsr.dll")
    print(f"[rtx_vsr] {out / 'rtxvsr.dll'} with nvngx_vsr.dll beside it", flush=True)


if __name__ == "__main__":
    main()
