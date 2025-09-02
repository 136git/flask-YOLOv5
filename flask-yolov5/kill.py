import os
def kill(pid):
    cmd = 'taskkill /pid ' + str(pid) + ' /f'

    try:
        os.system(cmd)
        print(pid, 'killed')

    except Exception as e:
        print(e)