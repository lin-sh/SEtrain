import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body):
    # 邮件配置
    sender_email = " "  # 发送者邮箱
    receiver_email = " "  # 接收者邮箱
    password = " "  # 发送者邮箱密码
    smtp_server = "smtp.126.com"  # 126 邮箱 SMTP 服务器
    smtp_port = 25  # 一般情况下使用 25 端口

    # 构造邮件内容
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # 连接到邮件服务器
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)

# 模型训练代码...

# 训练完成后发送邮件通知
# send_email("模型训练完成", "您的模型已经成功训练完成！")
