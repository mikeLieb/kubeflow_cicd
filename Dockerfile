FROM alpine:latest

COPY . .

CMD ["sh", "-c","cat /test/test.txt;echo hello"]
