在 Linux 中可以使用以下命令创建一个用户：

使用 root 权限创建用户
创建用户（需要 root 权限）

bash
复制代码
sudo useradd username  # 替换 'username' 为要创建的用户名
设置用户密码

bash
复制代码
sudo passwd username
创建 home 目录（在部分系统上，useradd 默认不创建 home 目录，可以使用 -m 参数确保创建）

bash
复制代码
sudo useradd -m username
限制 root 权限并使用普通用户的好处
使用普通用户而不是 root 用户的好处主要在于安全性和系统的稳定性：

安全性：

root 用户拥有系统的完全控制权限，任何错误操作（例如误删系统文件）都可能造成严重后果。普通用户的权限有限，即使误操作也不会轻易影响整个系统。
限制 root 权限可以降低系统被恶意攻击或病毒感染的风险。
防止误操作：

root 用户可以执行任何命令，而普通用户权限受到限制。以普通用户身份登录可以减少意外更改关键系统配置或删除系统文件的风险。
多用户管理：

在多用户环境中，使用普通用户账号可以为不同用户分配不同的权限，有助于资源管理和访问控制，确保用户之间的操作不会互相干扰。