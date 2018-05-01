function(input, output, session) {
  
  output$ratio_cumulatted_wins <- renderPlot({
    grafico_1 = historico_resultados_1%>% 
    arrange(., year,BackGround)%>%
    filter(., BackGround %in% input$checkGroup, year %in% seq(input$Period[1],input$Period[2], by=1))
    g <- ggplot(data = grafico_1, aes(x = year, y = ratio_wins_cumulated))
    g + geom_line(aes(color = BackGround))+labs(y = " ")+
    ggtitle(TeX('Win Ratio, accumulated. $\\frac{$\\sum_{} Wins$}{$\\sum_{} Fights$}$'))})


  output$quantidade_lutadores <- renderTable({
    auxiliar = historico_resultados_1%>%
      filter(., BackGround %in% input$checkGroup, year %in% seq(input$Period[1],input$Period[2], by=1))
    auxiliar = summarise(group_by(auxiliar, BackGround), 
                         qtd_fighters = sum(qtd_fighters),
                         wins = sum(wins), win_ratio = round(sum(wins)/sum(qtd_fighters),2)
                         ) %>% arrange(.,desc(win_ratio),qtd_fighters,wins)})

    output$winner_style <- renderPlot({
      aux = filter(ufc,winner_style%in%input$checkGroup)%>%
      filter(.,year %in% seq(input$Period[1],input$Period[2], by=1),
               f1BackGround %in% input$checkGroup|f2BackGround %in% input$checkGroup)
     
      historico_resultados_2 = data.frame(summarise(group_by(aux, winner_style), 
                                                    Amount_of_Fighters = sum(value),
                                                    FightDuration_minutes= mean(minutos_de_luta)))
    
      colnames(historico_resultados_2)[1] <- "style"
      ggplot(data =historico_resultados_2, aes(x = style, y = Amount_of_Fighters)) + 
      geom_bar(colour="black",stat = 'identity',aes(fill=style))+
      scale_fill_brewer(palette = "Blues")+labs(x = " ")+ggtitle("Winners Distribution") +
      scale_x_discrete(breaks=1:dim(historico_resultados_2)[1],
                       labels=rep(' ', dim(historico_resultados_2)[1]))})
  

  output$matriz <- renderTable({
    auxiliar = filter(ufc, f1BackGround %in% input$checkGroup, year %in% seq(input$Period[1],input$Period[2], by=1))
    auxiliar <- select( auxiliar, BackGround = f1BackGround, FinalDecision = method )
    auxiliar$value = 1
    matriz = data.frame(summarise(group_by(auxiliar, FinalDecision,BackGround),qtd = sum(value))) %>% spread(FinalDecision, qtd)
    matriz[is.na(matriz)] <- 0
   
    total_col = apply(matriz[,-1], 1, sum)
    pcts = lapply(matriz[,-1], function(x) {round(x / total_col,2)})
    pcts = as.data.frame(pcts)
    pcts$BackGround = matriz$BackGround
    pcts <- pcts %>%select(BackGround,everything())})
  
   output$leafletMap <- renderLeaflet({
     leaflet() %>% addTiles() %>% 
       addMarkers(lng=filter(gyms,style %in% input$checkGroup)$lng, lat=filter(gyms,style%in% input$checkGroup)$lat, popup=filter(gyms,style%in% input$checkGroup)$gymNameAdress)
     })
  
   observe({
     style <- unique(filter(styles,category == input$category)$style)
     updateSelectizeInput(session, "style",choices = style,selected = style[1])
     }) 
   
   observe({
     move <- unique(filter(moves,style == input$style)$move)
     updateSelectizeInput(session, "move",choices = move,selected = move[1])
     }) 
   
   output$tb <- renderUI({tags$video(src=paste(input$move,".mp4", sep = ""), 
                                     type = "video/mp4", controls = NA ,width="350px", height="350px")
     })
}
