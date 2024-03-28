export PATH=$PATH:/mnt/c/Windows/System32
go run . | dot -Tpng > output.png && wslview output.png
