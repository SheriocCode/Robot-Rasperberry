import os
import paramiko

def scp_transfer(local_file_path, remote_file_path, remote_host, remote_port, remote_username, remote_password):
    # 创建SSH对象
    ssh = paramiko.SSHClient()
    # 添加新的主机密钥策略（不推荐在生产环境中使用）
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None

    try:
        # 连接到SSH服务器
        ssh.connect(remote_host, port=remote_port, username=remote_username, password=remote_password)
        
        print('成功连接ssh')

        # 创建SFTP对象
        sftp = ssh.open_sftp()

        print('成功创建sftp')
        
        # 上传文件
        sftp.put(local_file_path, remote_file_path)
        
        print(f"文件 {local_file_path} 已成功上传到 {remote_file_path}")
        
    except Exception as e:
        print(f"传输文件时发生错误: {e}")
    finally:
        # 关闭连接
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()


print(f"当前工作目录: {os.getcwd()}")
# 使用示例
local_file_path = './agent_plan.txt'
# remote_file_path = '~/Code/agent_plan.txt'
remote_file_path = '/home/pi/Code/agent_plan.txt' # 使用绝对路径
remote_host = '192.168.149.1'
remote_port = 22
remote_username = 'pi'
remote_password = 'raspberry'

scp_transfer(local_file_path, remote_file_path, remote_host, remote_port, remote_username, remote_password)