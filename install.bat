@echo off

echo Installing dependencies...
echo.

rem 安装项目依赖
pip install -e ".[train]"
echo.

rem 安装其他依赖包
pip install ipdb tensorboard openai openpyxl datasets pytesseract decord mamba-ssm causal-conv1d
echo.

pip install -U pillow
echo.

pip install matplotlib yake
echo.

pip install open_clip_torch diffusers[torch] ezcolorlog
echo.

pip install timm==0.9.16
echo.

pip install opencv-python
echo.

pip install datasets pandas
echo.

pip install transformers qwen_vl_utils torch torchvision
echo.

rem 下载并安装flash attention
curl -o flash_attn_4-4.0.0b5-py3-none-any.whl https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta6/flash_attn_4-4.0.0b5-py3-none-any.whl
pip install flash_attn_4-4.0.0b5-py3-none-any.whl
echo.

rem 1. 强制升级 scikit-learn 到兼容版本（至少 1.4 以上，推荐最新）
rem 使用 --no-cache-dir 确保不使用旧缓存
pip install --upgrade --no-cache-dir scikit-learn
echo.

rem 2. 强制升级 numpy（可选，但推荐，因为很多新库需要 numpy >= 2.0）
rem 注意：这可能会导致某些旧库报错，但对于跑 Qwen 模型通常是必须的
pip install "numpy==2.1.3" --force-reinstall
echo.

echo Installation completed successfully!
pause