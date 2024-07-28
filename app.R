library(shiny)
library(plotly)
library(readr)
library(dplyr)
library(shinyjs)

# Define the UI
ui <- fluidPage(
  titlePanel("Gráfico Piper Interactivo"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Sube un archivo CSV", accept = ".csv"),
      uiOutput("color_by"),
      downloadButton("downloadPlot", "Descargar Gráfico")
    ),
    mainPanel(
      plotlyOutput("piperPlot")
    )
  ),
  useShinyjs()  # This is needed to update the color_by select input
)

# Define the server logic
server <- function(input, output, session) {

  # Reactive expression to read the data from the file
  data <- reactive({
    req(input$file)
    read_csv(input$file$datapath)
  })

  # Reactive expression to get column names for the 'color_by' selector
  columns <- reactive({
    df <- data()
    colnames(df)
  })

  # Update the 'color_by' selector options when columns are available
  output$color_by <- renderUI({
    req(columns())
    selectInput("color_by", "Agrupar por columna", choices = columns())
  })

  # Render the plotly plot
  output$piperPlot <- renderPlotly({
    df <- data()
    req(df)

    color_by <- input$color_by

    if (!is.null(color_by) && color_by %in% colnames(df)) {
      categories <- unique(df[[color_by]])
      p <- plot_ly()
      for (category in categories) {
        category_data <- df %>% filter(.[[color_by]] == category)
        p <- add_trace(p, x = category_data$Ca, y = category_data$Mg, 
                       type = 'scatter', mode = 'markers', 
                       name = category)
      }
    } else {
      p <- plot_ly(df, x = ~Ca, y = ~Mg, type = 'scatter', mode = 'markers')
    }

    p <- p %>% layout(title = "Piper Diagram", xaxis = list(title = "Ca"), yaxis = list(title = "Mg"))
    p
  })

  # Handle the download button
  output$downloadPlot <- downloadHandler(
    filename = function() { "Piper_plot.jpg" },
    content = function(file) {
      p <- output$piperPlot()
      if (is.null(p)) return(NULL)
      orca(p, file)  # Save the plot to a file using Orca
    }
  )
}

# Run the application
shinyApp(ui = ui, server = server)
