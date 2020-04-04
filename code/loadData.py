import csv
import os
import sys
# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
# Dask imports
import dask.bag as db
import dask.dataframe as df  # you can use Dask bags or dataframes
from csv import reader


#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

#Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: os.linesep.join([x,y]))
    return a + os.linesep

def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None



#Python answer functions
'''
def playOff_Teams():
   
    #This function will load all the teams that have made it to the playoffs
    #The data exists in the player_playoffs.     
    #player_playoffs.txt - playoff stats for all players    
    #team_season.txt - all teams statistics
    #The fields that will be chosen are year,team,leag
    #After loading the data the function will remover duplicates

    #Condition:
    #year > 1979  and Leag='N'
    
    #Return List[(Year,team,PlayOff)]
    #    Year: Number
    #    team: text
    #    PlayOff: 0 or 1
  
    # Get all teams that have played in the playoffs
    team_playOff_file="./data/player_playoffs.txt"
    spark = init_spark()
    df_playOff = spark.read.csv(team_playOff_file, header=True, mode="DROPMALFORMED", encoding='utf-8')
    df_playOff = df_playOff.select("year","team","leag").filter(df_playOff["year"].cast(IntegerType())>1979)
    df_playOff = df_playOff.filter(lit(df_playOff['leag'])=='N')
    df_playOff = df_playOff.withColumn("playOff",lit(1))
    df_playOff = df_playOff.distinct()
    df_playOff= df_playOff.sort("year",ascending=True)
    
    # Get all teams that have not played in the playoffs
    teams_file="./data/team_season.txt"
    df_teams = spark.read.csv(teams_file, header=True, mode="DROPMALFORMED", encoding='utf-8')
    df_teams = df_teams.select("year","team","leag").filter(df_teams["year"].cast(IntegerType())>1979)
    df_teams = df_teams.filter(lit(df_teams['leag'])=='N')
    df_teams = df_teams.select("year","team").subtract(df_playOff.select("year","team")).sort("year",ascending=True)
    df_teams = df_teams.withColumn("playOff",lit(0))
    print ("No play off " + str(df_teams.count()))
    print ( "Yes Play off " + str(df_playOff.count()))

    #Merge All teams in dataframe
    df = df_playOff.select("year","team","playOff").union(df_teams.select("year","team","playOff")).sort("year",ascending=True)
    
    print ( "All  " + str(df.count()))
    print(df.collect())

#a= playOff_Teams()
'''

def load_data():
    
    filename="./data/baskteball_reference_com_teams.csv"
    spark = init_spark()
    
    df = spark.read.csv(filename, header=True, mode="DROPMALFORMED", encoding='utf-8')    
    df = df.select("id","year","team","3P","2P","FT","TRB","AST","STL","BLK","TOV","PTS","MP","Playoff")
    df = df.withColumn("Points_Per_minute",col("PTS")/col("MP"))
    df = df.withColumn("3Points_Per_minute",col("3P")/col("MP"))
    df = df.withColumn("2Points_Per_minute",col("2P")/col("MP"))
    df = df.withColumn("FThrow_Per_minute",col("FT")/col("MP"))
    df = df.withColumn("Rebound_Per_minute",col("TRB")/col("MP"))
    df = df.withColumn("Assists_Per_minute",col("AST")/col("MP"))
    df = df.withColumn("Steals_Per_minute",col("STL")/col("MP"))
    df = df.withColumn("Blocks_Per_minute",col("BLK")/col("MP"))
    df = df.withColumn("TurnOvers_Per_minute",col("TOV")/col("MP"))
    

    data_classifiers = df.select("id","Playoff","Points_Per_minute","3Points_Per_minute","2Points_Per_minute","FThrow_Per_minute",
    "Rebound_Per_minute","Assists_Per_minute","Steals_Per_minute","Blocks_Per_minute","TurnOvers_Per_minute")
    
    return data_classifiers#.collect()

#a= load_data()