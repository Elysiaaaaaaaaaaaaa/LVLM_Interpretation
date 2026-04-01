sudo pip3 uninstall -y flash-attn timm

pip3 install -y -e ".[train]"
# pip3 install -y flash-attn --no-build-isolation
pip3 install -y ipdb tensorboard openai openpyxl datasets pytesseract decord mamba-ssm causal-conv1d
pip3 install -y -U pillow

pip3 install -y matplotlib yake
pip3 install -y open_clip_torch diffusers[torch] ezcolorlog
pip3 install -y timm==0.9.16
pip3 install -y opencv-python
pip3 install -y datasets pandas
