"""
Author: Karthikraja

QR code Program
"""
import qrcode 

data = 'Here is new QR code'

qr = qrcode.QRCode(version = 1, box_size=10, border=5)

qr.add_data(data)

qr.make(fit=True)
img = qr.make_image(fill_color = 'red', back_color = 'white')

img.save('./Python files/qrcode.png')
