#sphinx-apidoc -fPMe -o . ../src/ikats
sphinx-apidoc -fPe -o . ../src/ikats
make html

echo "you can open the doc with"
echo "google-chrome _build/html/index.html &"
