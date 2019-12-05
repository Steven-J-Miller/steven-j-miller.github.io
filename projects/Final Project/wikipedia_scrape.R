library(rvest)
library(magrittr)

getWeather <- function(url){
  webpage <- read_html(url)
  table <- html_table(webpage, header=FALSE, fill=TRUE)[[1]]
  dict <- as.data.frame(table)
  weather_words <- c("rain", "rainy")
  
  all_words <- unlist( # flattten word list from individual strings into one vector
    regmatches(dict$X2,  gregexpr('\\w+', dict$X2))) # extract all words
  
  #get frequency count of every word
  all_words <- sapply(all_words,tolower)
  freq_count <- as.data.frame(table(all_words))
  
  #get just the weather related terms
  weather_count <- data.frame(c("rain","rainy"), c(0,0))
  weather_count <- freq_count[freq_count$all_words %in% weather_words,]
  rain_count <- sum(weather_count$Freq)
  print(url)
  return(rain_count)
}
new_frame$url <- lapply(new_frame$url, as.character)
new_frame$rain_count <- sapply(new_frame$url, getWeather)
