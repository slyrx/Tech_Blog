
## How To Use
when uploading some change, you should wait for a little while, in order to let github refresh the jekyll server.

## issues
+ sometimes when you change the setting in **\_config.yml** , the page didn't show ang change. At this time, you should wait, or change some other True/False Setting to triggle the change updating.


## using docker as the test envirment
+ first get image
1. run docker app
2. docker pull docker pull jekyll/jekyll:3.8
3. docker run --name=jk -p 4000:4000 -it -v /Users/slyrx/slyrxStudio/github_good_projects/Tech_Blog:/srv/jekyll jekyll/jekyll:3.8 sh

+ rerun image
1. run docker app
2. docker ps -a
3. docker start -i 9949204b531a

## jeklly run
1. bundle update
2. bundle clean  
3. bundle exe jekyll build
4. bundle exec jekyll serve -H 0.0.0.0 -P 4000 -I
