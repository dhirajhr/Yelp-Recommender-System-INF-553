import java.io.{File, PrintWriter}
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import collection.mutable.HashMap
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object ModelBasedCF {
  def main(args : Array[String]): Unit = {
    val t1 = System.currentTimeMillis()
    val start_time = System.nanoTime
    //val writer = new PrintWriter("src/main/resources/files_res.txt")
    val writer = new PrintWriter("Dhiraj_Ramnani_ModelBasedCF.txt")
    var conf = new SparkConf().setAppName("StackOverflow").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("ERROR")


   //val train_csv = sc.textFile("src/main/resources/train_review.csv")
   val train_csv = sc.textFile(args(0))
    val train_headerAndRows = train_csv.map(line => line.split(",").map(_.trim))
    val train_header = train_headerAndRows.first
    val train_rdd = train_headerAndRows.filter(_ (0) != train_header(0))

    //val test_csv = sc.textFile("src/main/resources/test_review.csv")
    val test_csv = sc.textFile(args(1))
    val test_headerAndRows = test_csv.map(line => line.split(",").map(_.trim))
    val test_header = test_headerAndRows.first
    val test_rdd = test_headerAndRows.filter(_ (0) != test_header(0))
    train_rdd.cache()
    test_rdd.cache()

    var userMap = new HashMap[String, Int]()
    var itemMap = new HashMap[String, Int]()
    var userMap_rev = new HashMap[Int, String]()
    var itemMap_rev = new HashMap[Int, String]()
    //id gen
    var item_increment = 1
    var user_increment = 1
    val test_format_gen = test_rdd.map { case Array(user: String, item: String, ratings: String) => (item, user) }
    val train_format_gen = train_rdd.map { case Array(user: String, item: String, ratings: String) => (item, user) }

    train_format_gen.collect().foreach { tuple =>
      if (!userMap.contains(tuple._2)) {
        userMap.put(tuple._2, user_increment)
        userMap_rev.put(user_increment, tuple._2)
        user_increment += 1
      }
      if (!itemMap.contains(tuple._1)) {
        itemMap.put(tuple._1, item_increment)
        itemMap_rev.put(item_increment, tuple._1)
        item_increment += 1
      }
    }

    test_format_gen.collect().foreach { tuple =>
      if (!userMap.contains(tuple._2)) {
        userMap.put(tuple._2, user_increment)
        userMap_rev.put(user_increment, tuple._2)
        //userMap1.put(user_increment, tuple._2)
        user_increment += 1
      }
      if (!itemMap.contains(tuple._1)) {
        itemMap.put(tuple._1, item_increment)
        itemMap_rev.put(item_increment, tuple._1)
        //itemMap1.put(item_increment, tuple._1)
        item_increment += 1
      }
    }



    val ratings = train_rdd.map{ case Array(user_id:String, business_id:String, stars:String) =>
      Rating(userMap(user_id), itemMap(business_id), stars.toDouble)
    }
    val ratings_test = test_rdd.map{ case Array(user_id:String, business_id:String, stars:String) =>
      Rating(userMap(user_id), itemMap(business_id), stars.toDouble)
    }

    val rank = 2
    val numIterations = 20
    val model = ALS.train(ratings, rank, numIterations, 0.223,1,1L)

    val usersProduct_test = ratings_test.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions =
      model.predict(usersProduct_test).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.sortByKey()




    val file_write =predictions.map(a=>userMap_rev(a._1._1)+","+itemMap_rev(a._1._2)+","+a._2)
    val list = file_write.collect().toList
    list.foreach(writer.write)
    writer.close()


    val ratesAndPreds = ratings_test.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)


    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    var rmse=Math.pow(MSE,0.5)
    val levels = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = Math.abs(r1 - r2)
      err
    }
    val t2 = System.currentTimeMillis()
    var one=0
    var two=0
    var three=0
    var four=0
    var five=0
    for(x<- levels.collect() ){
      if (x>=0 && x<1)
        one+=1

      if (x>=1 && x<2)
        two+=1

      if (x>=2 && x<3)
        three+=1

      if (x>=3 && x<4)
        four+=1

      if (x>=4)
        five+=1

    }
    println(">=0 and <1: "+one)
    println(">=1 and <2: "+two)
    println(">=2 and <3: "+three)
    println(">=3 and <4: "+four)
    println(">=4: "+five)
    println("RMSE: "+rmse)
    println("Time: "+(t2-t1)/1000+" sec")

  }

}
