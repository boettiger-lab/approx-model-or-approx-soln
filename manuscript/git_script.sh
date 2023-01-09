pandoc -o manuscript.pdf manuscript.Rmd
git commit -am "$1"
git pull
git push