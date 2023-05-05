# make checkpoints directory
mkdir checkpoints
cd checkpoints

# downlaod 0.3B
git lfs install
git clone https://www.modelscope.cn/damo/ChatPLUG-240M.git

# download 3.7B
git lfs install
git clone https://www.modelscope.cn/damo/ChatPLUG-3.7B.git
