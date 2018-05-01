library(shiny)
library(leaflet)
library(dplyr)
library(ggplot2)
library(latex2exp)
library(tidyr) 


address = '/Users/felipesantos/Desktop/BootcampNY/shiny/martial_arts_app'
#address = '.'

gyms_ = paste(address,'data_gyms.csv',sep =  '/')
gyms <- read.csv(gyms_, stringsAsFactors = F, sep = ";")
colnames(gyms)[1] <- "style"

moves_ = paste(address,'data_moves.csv',sep =  '/')
moves <- read.csv(moves_, stringsAsFactors = F, sep = ";")
colnames(moves)[1] <- "style"


styles_ = paste(address,'data_styles.csv',sep =  '/')
styles <- read.csv(styles_, stringsAsFactors = F, sep = ";")
colnames(styles)[1] <- "category"

ufc_definitiva = paste(address,'ufc_definitiva.csv',sep =  '/')
ufc <- read.csv(ufc_definitiva, stringsAsFactors = F)

historico_resultados = paste(address,'historico_resultados_1.csv',sep =  '/')
historico_resultados_1 <- read.csv(historico_resultados, stringsAsFactors = F)

historico_resultados_1$value = 1
artes = data.frame(summarise(group_by(historico_resultados_1, BackGround), qtd_fighters = sum(value))
                   %>%arrange(., desc(qtd_fighters)))$BackGround

fim = max(historico_resultados_1$year)
inicio = min(historico_resultados_1$year)

