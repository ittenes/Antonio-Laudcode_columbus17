import socket

host='192.168.0.100'
port=8080

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host,port))
s.listen(1)
conn, addr = s.accept()

string = ''

while True:
    d = conn.recv(640*480)

    if not d:
        break

    else:
        d = d.decode('UTF-8')
        string += d

print (string)
print (len(string))

fh = open("imageToSave.jpeg", "wb")
fh.write(string)
fh.close()
