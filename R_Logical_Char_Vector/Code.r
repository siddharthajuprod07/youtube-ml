### logical Vectors
x <- c(10,5,15,6,25)
t = x > 10
j = x <=15
# Operators <, <=, >, >=, ==, !=
k = j & t # logical AND 
l = j | t # logical OR
m = !j

### Character Vectors
?Quotes4
paste(c("hello","world"),1:10)
paste(c("hello","world"),1:10,sep = "$")
i = c("x1","y1")
j = c("x2","y2")
k1 = c(i,j)
k2 = paste(i,j)
nth <- paste0(1:12, c("st", "nd", "rd", rep("th", 9)))