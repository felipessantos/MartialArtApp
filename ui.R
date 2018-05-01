fluidPage(
  titlePanel("UFC analysis"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("Period", label = h3("Year Range"), 
                  round = T,format = "####",step = 1,
                  min = inicio, max = fim, value = c(inicio,fim)
                  ),
      checkboxGroupInput("checkGroup", label = h3("Choose the Style"), 
                         choices = artes,
                         selected = c('Judo','Taekwondo', 'Karate', 'Kickboxing', 
                                      'Boxing', 'Muay Thai', 'Wrestling', 
                                      'Brazilian Jiu Jitsu')
                         ),hr()
    ),
    mainPanel(
      tableOutput("quantidade_lutadores"),
      plotOutput("ratio_cumulatted_wins"),
      plotOutput("winner_style"),
      tableOutput("matriz"),
      h4("\n where to train...   Gyms in Mahatan"),
      leafletOutput("leafletMap",height = 400, width = 600) ,

      h4("\n  A few moves, from different arts"),
      textOutput("moves"),
      
      selectizeInput(inputId = "category",
                     label = "Choose category",
                     choices = unique(styles$category)),
      selectizeInput(inputId = "style",
                     label = "Choose style",
                     choices = unique(styles$style)),
      selectizeInput(inputId = "move",
                     label = "Choose move",
                     choices = unique(moves$move)),
      uiOutput("tb")
    )
  )
)

