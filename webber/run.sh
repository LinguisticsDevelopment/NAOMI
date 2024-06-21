set -xe

go run . | dot -Tpng > output.png

open output.png
