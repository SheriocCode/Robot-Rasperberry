import paramiko

def ssh_execute_command(remote_host, remote_port, remote_username, remote_password, command):
    # 创建SSH对象
    ssh = paramiko.SSHClient()
    # 添加新的主机密钥策略（不推荐在生产环境中使用）
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 连接到SSH服务器
        ssh.connect(remote_host, port=remote_port, username=remote_username, password=remote_password)
        
        print('成功连接ssh')

        # 执行远程命令
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # 获取命令输出
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        # 打印输出和错误信息
        if output:
            print(f"命令输出:\n{output}")
        if error:
            print(f"错误信息:\n{error}")
        
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
    finally:
        # 关闭连接
        if ssh:
            ssh.close()

# 使用示例
remote_host = '192.168.31.146'
remote_port = 22
remote_username = 'pi'
remote_password = 'raspberry'
command = 'python ~/TonyPi/OpenVINO/utils_robot.py'

ssh_execute_command(remote_host, remote_port, remote_username, remote_password, command)