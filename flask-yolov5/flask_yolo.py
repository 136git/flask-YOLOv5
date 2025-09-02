import flask
from flask import Flask,render_template,request,send_from_directory, render_template
import os
import time
import psutil
import pymysql
import hashlib
from PIL import Image
import subprocess
import win32api,win32con
import threading
from random import choice
from multiprocessing import Process

#  创建应用程序
#  web应用程序
# 创建Flask程序并定义模板位置
app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates'
            )
basedir = os.path.abspath(os.path.dirname(__file__))

url = ''
num = 0
root_dir = r'./static/result/date'

@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.redirect(flask.url_for('log_in'))

login = False
@app.route('/log_handle', methods=['POST']) #处理登录并判段是否为管理员
def log_handle():
    root = False
    find_user = False
    if request.method == 'POST':
        # username和password是前端log_in.html的name字段里的字符
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'root' and password == '123456dog':
            root = True
        # 对密码进行md5处理
        # encrypass = hashlib.md5()
        # encrypass.update(password.encode(encoding='utf-8'))
        # password = encrypass.hexdigest()

    # 通过mysql进行存储
    db = pymysql.connect(host="127.0.0.1",
                         port=3306,
                         user="root",
                         password="123456dog",
                         db="test1")

    # 创建数据库指针cursor
    cursor = db.cursor()

    sql = "SELECT * FROM login"

    # 执行数据库命令并将数据提取到cursor中
    cursor.execute(sql)
    # 确认命令
    db.commit()
    data = cursor.fetchall()
    user_list = []
    username_list = []
    for item in data:
        dict_user = {'username': item[0], 'password': item[1]}
        show_user = item[0]
        print(item[1])
        user_list.append(dict_user)
        username_list.append(show_user)
    # 对数据库中所有的数据进行遍历,找出username
    for i in range(len(user_list)):
        if user_list[i]['username'] == username:
            if user_list[i]['password'] == password:
                find_user = True
                break
            else:
                break
    db.close()
    global login
    if root:
        login = True
        return flask.render_template("root.html", username=username,username_list=username_list,data=data)

        # 登录失败就跳转倒log_fail中并弹窗
    elif not find_user:
        return flask.render_template("log_fail.html")
    else:
        # 登录成功就跳转log_success,并将用户名带入
        login = True
        return flask.render_template("log_success.html", username=username)


#处理注销
@app.route('/log_off_handle', methods=['POST'])
def log_off_handle():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
            # 对密码进行md5处理
        # encrypass = hashlib.md5()
        # encrypass.update(password.encode(encoding='utf-8'))
        # password = encrypass.hexdigest()

        db = pymysql.connect(host="127.0.0.1",
                             user="root",
                             password="123456dog",
                             port=3306,
                             db="test1")
        cursor = db.cursor()
        search_sql = "SELECT * FROM login"
        cursor.execute(search_sql)
        db.commit()
        user_list = []
        data = cursor.fetchall()
        for item in cursor.fetchall():
            dict_user = {'username': item[0], 'password': item[1]}
            user_list.append(dict_user)

        exist = False
        for i in range(len(user_list)):
            if user_list[i]['username'] == username:
                if user_list[i]['password'] == username:
                    exist = True
                # 将用户名从数据库中删除
        if exist == True:
            sql = "delete from login where username=%s"
            cursor.execute(sql,(username))
            db.commit()
        else:
            log_off_fail = 1
            return flask.render_template("log_off_fail.html", log_off_fail=log_off_fail)
    db.close()
    log_off_success =1

    return flask.render_template("log_off_success.html",log_off_success=log_off_success)

# 处理注册
@app.route('/register_handle', methods=['POST'])
def register_handle():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        # 判断两次密码是否正确
        if password == confirm_password:
            # 对密码进行md5处理
            # encrypass = hashlib.md5()
            # encrypass.update(password.encode(encoding='utf-8'))
            # password = encrypass.hexdigest()

            db = pymysql.connect(host="127.0.0.1",
                                 user="root",
                                 password="123456dog",
                                 port=3306,
                                 db="test1")
            cursor = db.cursor()

            search_sql = "SELECT * FROM login"
            cursor.execute(search_sql)
            db.commit()
            user_list = []
            for item in cursor.fetchall():
                dict_user = {'username': item[0], 'password': item[1]}
                user_list.append(dict_user)

            repeat = False
            for i in range(len(user_list)):
                # 判断是否存在相同用户名
                if user_list[i]['username'] == username:
                    print(i)
                    repeat = True
                    # 将用户名和加密后的密码插入数据库
            if repeat == False:
                sql = "INSERT INTO login VALUES('%s','%s')" % (username, password)
                cursor.execute(sql)
                db.commit()
            else:
                have_same_username = 1
                return flask.render_template("register_fail.html", have_same_username=have_same_username)
        else:
            two_passwd_wrong = 1
            return flask.render_template("register_fail.html", two_passwd_wrong=two_passwd_wrong)
    db.close()
    register_success =1
    return flask.render_template("register_success.html",register_success=register_success)

@app.route('/root_register', methods=['POST'])##管理员注册
def root_register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        db = pymysql.connect(host="127.0.0.1",
                             user="root",
                             password="123456dog",
                             port=3306,
                             db="test1")
        cursor = db.cursor()
        search_sql = "SELECT * FROM login"
        # 判断两次密码是否正确
        if password == confirm_password:

            cursor.execute(search_sql)
            db.commit()
            user_list = []
            data = cursor.fetchall()

            for item in data:
                dict_user = {'username': item[0], 'password': item[1]}
                user_list.append(dict_user)

            repeat = False
            for i in range(len(user_list)):
                # 判断是否存在相同用户名
                if user_list[i]['username'] == username:
                    print(i)
                    repeat = True
                    # 将用户名和加密后的密码插入数据库
            if repeat == False:
                sql = "INSERT INTO login VALUES('%s','%s')" % (username, password)
                cursor.execute(sql)
                db.commit()
            else:
                have_same_username = 1
                return flask.render_template("root_register_fail.html", have_same_username=have_same_username,data=data)
        else:
            cursor.execute(search_sql)
            db.commit()
            data = cursor.fetchall()
            two_passwd_wrong = 1
            return flask.render_template("root_register_fail.html", two_passwd_wrong=two_passwd_wrong,data=data)
    cursor.execute(search_sql)
    db.commit()
    data = cursor.fetchall()
    db.close()
    login = True
    register_success =1
    return flask.render_template("root_register_success.html",register_success=register_success,data=data)
#处理管理员删除
@app.route('/del_handle', methods=['POST'])
def del_handle():
    global login

    db = pymysql.connect(host="127.0.0.1",
                         user="root",
                         password="123456dog",
                         port=3306,
                         db="test1")
    cursor = db.cursor()
    search_sql = "SELECT * FROM login"
    if request.method == 'POST':
        username = request.form.get('value')
        cursor.execute(search_sql)
        db.commit()
        user_list = []
        username_list = []
        data = cursor.fetchall()
        for item in data:
            dict_user = {'username': item[0]}
            show_user = item[0]
            user_list.append(dict_user)
            if show_user!='root':
                username_list.append(show_user)
        exist = False
        if username!='root':
            for i in range(len(user_list)):
                if user_list[i]['username'] == username:
                    exist = True
                    # 将用户名从数据库中删除
        if exist == True:
            sql = "delete from login where username=%s"
            cursor.execute(sql,(username))
            db.commit()
        else:
            del_fail = 1
            return flask.render_template("root_del_fail.html", del_fail=del_fail,username_list=username_list,data=data)
    cursor.execute(search_sql)
    db.commit()
    data = cursor.fetchall()
    login = True
    db.close()
    del_success =1
    return flask.render_template("root_del_success.html",del_success=del_success,username_list=username_list,data=data)


@app.route('/test',methods=['post'])
def test():
    if request.method == 'POST':
        username = request.form.get('value')
        password = request.form.get('aaa')
        print(username)
        print(password)
        return 'username'
#处理管理员修密码
@app.route('/reset_handle', methods=['POST'])
def reset_handle():
    global login

    db = pymysql.connect(host="127.0.0.1",
                         user="root",
                         password="123456dog",
                         port=3306,
                         db="test1")
    cursor = db.cursor()
    search_sql = "SELECT * FROM login"
    if request.method == 'POST':
        username = request.form.get('value')
        password = request.form.get('repassword')

        cursor.execute(search_sql)
        data = cursor.fetchall()
        db.commit()
        user_list = []
        username_list = []
        for item in data:
            dict_user = {'username': item[0],'password': item[1]}
            show_user = item[0]
            user_list.append(dict_user)
            if show_user!='root':
                username_list.append(show_user)
        exist = False
        if username!='root':
            for i in range(len(user_list)):
                if user_list[i]['username'] == username:
                    exist = True
        if exist == True:
            sql = "update login set password=%s where username=%s"
            # print(password)
            cursor.execute(sql,(password,username))
            db.commit()
        else:
            reset_fail = 1
            return flask.render_template("root_reset_fail.html", reset_fail=reset_fail,username_list=username_list,data=data)
    cursor.execute(search_sql)
    data = cursor.fetchall()
    db.commit()
    db.close()
    reset_success =1
    login = True
    return flask.render_template("root_reset_success.html",reset_success=reset_success,data=data)

@app.route('/picture', methods=['post'])
def picture():
    return flask.render_template('picture.html')

@app.route('/web', methods=['post'])
def web():
    return flask.render_template('web.html')

@app.route('/source', methods=['post'])
def source():
    return flask.render_template('source.html')

@app.route('/log_in', methods=['GET'])
def log_in():
    return render_template('log_in.html')

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/log_off',methods=['GET'])
def log_off():
    return render_template('log_off.html')


# 自定义404页面
@app.errorhandler(404)
def page_not_found(error):
    return flask.render_template("404.html"), 404


@app.route("/number")
def number():
    global login
    global num
    num = 0
    print(num)
    return render_template('abc.html',img='005.webp')
@app.route("/welcome",methods=['POST'])
def indexer():
    global login
    global num
    # num = 0
    # print(num)
    if login==True:
        return flask.redirect(flask.url_for('number'))
    else:
        return render_template('log_in.html')

@app.route("/url_post",methods=["POST"]) #网页链接图片识别
def url_train():
    global login
    login=True
    url = request.form.get("url")
    cmd = r'python .\yolov5_master\picture.py --web ' \
          + '"' + url + '"' + \
          " --name ./static/result --imgsz 640 --weights ./yolov5_master/yolov5_weights/14.pt "
    text = os.popen(cmd)
    print(text.read())
    print(url)
    f1 = open(file='cord.txt', mode='r')
    pic = f1.read()
    f1.close()
    if pic == '1':
        cord = '检测到狗栓了狗绳！'
    else:
        cord = '检测到有狗没栓狗绳！'
    print(pic)
    return render_template("web.html",temp='res.jpg',cord=cord)

# 表单提交路径，需要指定接受方式
@app.route('/getImg', methods=['GET', 'POST']) #上传图片
def getImg():
    global login
    login=True
    # 通过表单中name值获取图片
    imgData = request.files["image"]
    # print(imgData)
    # 设置图片要保存到的路径
    path = basedir + "/static/images/"
    # 获取图片名称及后缀名
    imgName = imgData.filename
    # 图片path和名称组成图片的保存路径
    file_path = path + imgName
    # 保存图片
    imgData.save(file_path)
    # url是图片的路径
    url = './static/images/' + imgName
    cmd = r'python .\yolov5_master\picture.py --source ' + url  + " --name ./static/result --imgsz 640 --weights ./yolov5_master/yolov5_weights/14.pt"
    text = os.popen(cmd)
    print(text.read())
    f1 = open(file='cord.txt', mode='r')
    pic = f1.read()
    f1.close()
    if pic == '1':
        cord = '检测到狗栓了狗绳！'
    else:
        cord = '检测到有狗没栓狗绳！'
    # print(url)
    return render_template('picture.html',img_path=imgName,cord=cord)

@app.route('/show_img', methods=['get','POST'])
def shower():
    show = request.form.get("show")
    path = "./static/result/result.txt"
    data = ""
    with open(path, encoding='utf-8') as filename:
        for line in filename:
            data = data + line.rstrip() + '\n'
    return render_template('show.html',show=show,data=data)

@app.route("/reset",methods=["POST"])
def renew():
    global num,stoppid
    num = 0
    stoppid = 0
    return render_template("login.html",img="005.webp")
@app.route('/random_image',methods=['post'])
def show_list():
    files = os.listdir(root_dir)
    isdir_list = gen_isdir_list(root_dir)
    global num
    global url
    num = num + 1
    url = request.form.get("url")
    def cmd():
        cmd = r'python .\yolov5_master\finally.py --source ' +\
              '"' + url + '"' + \
              " --name ./static/result --classes 0 1 2"
        text = os.popen(cmd)
        print(text.read())
    if num == 1:
        t = threading.Thread(target=cmd, args=())
        print(threading)
        t.start()
        print(1)
    else:
        print(num)
    return render_template("files_list.html", files=files, isdir_list=isdir_list)

@app.route('/stop', methods=['get','POST'])
def pid_kill():
    from kill import kill
    global num
    num = 0
    stoppid=0
    print(stoppid)
    stoppid+=1
    if stoppid ==1:
        f1 = open(file='pid.txt', mode='r')
        pid = f1.read()
        f1.close()
        kill(pid=pid)
        # print('kill')
        with open("pid.txt", 'w') as file:
            file.truncate(0)
    files = os.listdir(root_dir)
    isdir_list = gen_isdir_list(root_dir)
    word = '后台检测进程已结束！'
    return render_template("files_list.html", files=files, isdir_list=isdir_list,word=word,img='005.webp')

@app.route('/stop_return', methods=['get','POST'])
def stop_return():
    from kill import kill
    global num
    num = 0
    stoppid=0
    print(stoppid)
    stoppid+=1
    if stoppid ==1:
        f1 = open(file='pid.txt', mode='r')
        pid = f1.read()
        f1.close()
        kill(pid=pid)
        # print('kill')
        with open("pid.txt", 'w') as file:
            file.truncate(0)
    files = os.listdir(root_dir)
    isdir_list = gen_isdir_list(root_dir)
    word = '后台检测进程已结束！'
    return render_template("abc.html", files=files, isdir_list=isdir_list,word=word,img='005.webp')

@app.route('/<path:sub_dir>')
def sub_dir1_page(sub_dir):
    dir_name = root_dir + '\\' + sub_dir
    files = os.listdir(dir_name)
    isdir_list = gen_isdir_list(dir_name)
    return render_template("files_list.html", files=files, isdir_list=isdir_list)


@app.route('/<path:sub_dir1>/<path:sub_dir2>')
def sub_dir2_page(sub_dir1, sub_dir2):
    dir_name = root_dir + '\\' + sub_dir1 + '\\' + sub_dir2
    files = os.listdir(dir_name)
    isdir_list = gen_isdir_list(dir_name)
    return render_template("files_list.html", files=files, isdir_list=isdir_list)


@app.route('/<filename>')
def download_root(filename):
    return send_from_directory(root_dir, filename)


@app.route('/<path:sub_dir>/<filename>')
def download_subdir1(sub_dir, filename):
    dir_name = root_dir + '\\' + sub_dir
    return send_from_directory(dir_name, filename)


@app.route('/<path:sub_dir1>/<path:sub_dir2>/<filename>')
def download_subdir2(sub_dir1, sub_dir2, filename):
    dir_name = root_dir + '\\' + sub_dir1 + '\\' + sub_dir2
    return send_from_directory(dir_name, filename)


def gen_isdir_list(dir_name):
    files = os.listdir(dir_name)
    isdir_list = []
    for f in files:
        if os.path.isdir(dir_name + '\\' + f):
            isdir_list.append(True)
        else:
            isdir_list.append(False)
    return isdir_list

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True)  #启动应用程序->启动一个flask项目

#模板 ->html




