import tarfile
import os

def create_tar_gz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=os.path.relpath(file_path, source_dir))

# 指定要创建的tar.gz文件名和源目录
output_filename = "E:/PythonProject/mi_hutb.tar.gz"
source_dir = "E:/PythonProject/mi_hutb"

# 创建tar.gz文件
create_tar_gz(output_filename, source_dir)
