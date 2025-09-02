# -*- coding: utf-8 -*-
import base64

png = open('res.jpg','rb')
res = png.read()
s = base64.b64encode(res)
png.close()
print(s.decode('ascii'))

