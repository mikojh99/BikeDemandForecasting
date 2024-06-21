using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.SqlClient;
using System.Reflection;


string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
string modelPath = Path.Combine(rootDir, "MLModel.zip");
var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";

MLContext mlContext = new MLContext();

//DatabaseLoader that loads records of type ModelInput
DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();

//Define the query to load the data from the database
string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

//Create a DatabaseSource to connect to the database and execute the query
DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                connectionString,
                                query);
//Load the data into an IDataView
IDataView dataView = loader.Load(dbSource);


IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

//Define a pipeline that uses the SsaForecastingEstimator to forecast values in a time-series dataset

var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");



SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

Evaluate(secondYearData, forecaster, mlContext);

var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
forecastEngine.CheckPoint(mlContext, modelPath);

Forecast(secondYearData, 7, forecastEngine, mlContext);

void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{

    // Make predictions
    IDataView predictions = model.Transform(testData);

    // Actual values
    IEnumerable<float> actual =
    mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
        .Select(observed => observed.TotalRentals);

    // Predicted values
    IEnumerable<float> forecast =
    mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
        .Select(prediction => prediction.ForecastedRentals[0]);

    // Calculate error (actual - forecast)
    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    // Get metric averages
    var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
    var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

    // Output metrics
    Console.WriteLine("Evaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
    Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");


}

void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
{

    ModelOutput forecast = forecaster.Predict();

    IEnumerable<string> forecastOutput =
    mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
        .Take(horizon)
        .Select((ModelInput rental, int index) =>
        {
            string rentalDate = rental.RentalDate.ToShortDateString();
            float actualRentals = rental.TotalRentals;
            float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
            float estimate = forecast.ForecastedRentals[index];
            float upperEstimate = forecast.UpperBoundRentals[index];
            return $"Date: {rentalDate}\n" +
            $"Actual Rentals: {actualRentals}\n" +
            $"Lower Estimate: {lowerEstimate}\n" +
            $"Forecast: {estimate}\n" +
            $"Upper Estimate: {upperEstimate}\n";
        });

    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");
    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}



public class ModelInput
{
    //RentalDate: The date of the observation
    public DateTime RentalDate { get; set; }
    //Year: The encoded year of the observation (0=2011, 1=2012)
    public float Year { get; set; }
    //TotalRentals: The total number of bike rentals for that day
    public float TotalRentals { get; set; }
}

public class ModelOutput
{
    //ForecastedRentals: The predicted values for the forecasted period
    public float[] ForecastedRentals { get; set; }
    //LowerBoundRentals: The predicted minimum values for the forecasted period
    public float[] LowerBoundRentals { get; set; }
    //UpperBoundRentals: The predicted maximum values for the forecasted period
    public float[] UpperBoundRentals { get; set; }
}

