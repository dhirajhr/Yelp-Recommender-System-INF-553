import java.io.{File, PrintWriter}
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import collection.mutable.HashMap

object ItemBasedCF {
  def main(args : Array[String]): Unit = {

    val start_time = System.nanoTime
    var conf = new SparkConf().setAppName("StackOverflow").setMaster("local[*]")
    val sc = new SparkContext(conf)
    //conf.set("spark.executor.memory","4g")
    //conf.set("spark.memory.fraction","0.8")
    //conf.set("spark.executor.extraJavaOptions","-XX:-UseGCOverheadLimit")
    sc.setLogLevel("ERROR")

    //Training Data

    //val train_csv = sc.textFile("src/main/resources/train_review.csv")
    val train_csv = sc.textFile(args(0))
    val train_headerAndRows = train_csv.map(line => line.split(",").map(_.trim))
    val train_header = train_headerAndRows.first
    val train_rdd = train_headerAndRows.filter(_(0)!=train_header(0)).sample(false,0.6,1L)

    //Testing Data

    //val test_csv = sc.textFile("src/main/resources/test_review.csv")
    val test_csv = sc.textFile(args(1))
    val test_headerAndRows = test_csv.map(line => line.split(",").map(_.trim))
    val test_header = test_headerAndRows.first
    val test_rdd = test_headerAndRows.filter(_(0)!=test_header(0))
   // train_rdd.cache()
    //test_rdd.cache()

    //Forward and reverse dictionaries

    var userMap = new HashMap[String,Int]()
    var itemMap = new HashMap[String,Int]()
    var userMap_rev = new HashMap[Int,String]()
    var itemMap_rev = new HashMap[Int,String]()
    //id gen
    var item_increment = 1
    var user_increment = 1
    val test_format_gen = test_rdd.map{ case Array(user:String,item:String,ratings:String)=>(item,user) }
    val train_format_gen = train_rdd.map{ case Array(user:String,item:String,ratings:String)=>(item,user) }

    train_format_gen.collect().foreach{tuple=>
      if(!userMap.contains(tuple._2)) {
        userMap.put(tuple._2, user_increment)
        userMap_rev.put(user_increment, tuple._2)
        user_increment += 1
      }
      if(!itemMap.contains(tuple._1)) {
        itemMap.put(tuple._1, item_increment)
        itemMap_rev.put(item_increment, tuple._1)
        item_increment += 1
      }
    }

    test_format_gen.collect().foreach{tuple=>
          if(!userMap.contains(tuple._2)) {
            userMap.put(tuple._2, user_increment)
            userMap_rev.put(user_increment, tuple._2)
            user_increment += 1
          }
        if(!itemMap.contains(tuple._1)) {
          itemMap.put(tuple._1, item_increment)
          itemMap_rev.put(item_increment, tuple._1)
          item_increment += 1
        }
      }


    //id gen

    val item_avg_inter = train_rdd.map{ case Array(user:String,item:String,ratings:String)=>(itemMap.get(item).get,ratings.toDouble) }.groupByKey()
    val item_avg =  item_avg_inter.map(tuple=>(tuple._1,tuple._2.sum / tuple._2.size ))
    val all_avg =  item_avg_inter.flatMap{tup=> tup._2}
    val average_total = all_avg.sum()/all_avg.count()
    val avg_lookup = item_avg.collectAsMap()

    val rdd_item = train_rdd.map{ case Array(user:String,item:String,ratings:String)=>(itemMap.get(item).get,(userMap.get(user).get,ratings.toDouble)) }



    val norm = rdd_item.map{case (item:Int,(user:Int,ratings:Double))=>
      (user,(item,ratings-avg_lookup.get(item).get))}
    //norm.foreach(println)
    //norm.cache()


    //train: item,(user,rating)
    val test_format = test_rdd.map{ case Array(user:String,item:String,ratings:String)=>(itemMap.get(item).get,1) }.reduceByKey(_+_)
    val norm_format = norm.map{ case (user:Int,(item:Int,ratings:Double))=>(item,(user,ratings)) }


    val joined = norm_format.join(test_format)

    val joined_format =  joined.map{case (item,((user_train,rating_train),count))=>
      (user_train,(item,rating_train))
    }
    //joined_format.cache()



    val item_map = norm.map{ case (user:Int,(item:Int,ratings))=>(item,ratings*ratings)}.reduceByKey(_+_).map{
      case (item,rating)=>(item,Math.sqrt(rating))
    }.collectAsMap()

    val joined_norm = norm.join(joined_format)
    val item_pairs = joined_norm.map{ case (user:Int,((item1:Int,rating1),(item2:Int,rating2)))=>((item1:Int,item2:Int),(rating1,rating2))}
      .filter(tuple=>tuple._1._1<tuple._1._2)
    //item_pairs.cache()

    val numerator = item_pairs.map(tuple=> (tuple._1,tuple._2._2*tuple._2._1)).reduceByKey(_+_)
    val deno = item_pairs.map(tuple=> (tuple._1,(item_map.get(tuple._1._1.toInt).get)*(item_map.get(tuple._1._2.toInt).get)))
    val weight = numerator.join(deno).map{
      case ((useri,userj),(num:Double,denom:Double)) =>
        if(denom ==0)
          ((useri,userj),0.0)
        else
          ((useri,userj),num/denom)
    }

    val weight_updated = weight.map{case ((item1:Int,item2:Int),rating:Double)=>(item1,(item2,rating.toDouble))}
      .groupByKey().map({ case (k, numbers) => k -> numbers.toList.sortBy(_._2).reverse.take(2)} )
      .flatMap({case (k, numbers) => numbers.map(k -> _)}).map{case (item1,(item2,rating))=>((item1,item2),rating)}
   //weight_updated.cache()


    //PREDICTION
    val test_pred = test_rdd.map{ case Array(user:String,item:String,ratings:String)=>(userMap.get(user).get,(itemMap.get(item).get,ratings.toDouble)) }
    val user_join = test_pred.join(norm)


    val user_join_map1 = user_join.map { case (user, ((item_test, rating_test), (item_train, rating_train))) =>
      if(item_train < item_test){
      ((item_train, item_test), (user, item_test, rating_train))

      }
        else{
        ((item_test, item_train), (user, item_test, rating_train))}
    }.filter(tuple=>tuple._1._1!=tuple._1._2).join(weight_updated).map{ case ((item_train, item_test_temp), ((user, item_test, rating_train),wt))=>

      ((user,item_test),(wt,rating_train))
    }

    //Denominator for prediction

    val den_predict = user_join_map1.map{ case((user,item_test),(wt,rating_train))=>

      ((user,item_test),Math.abs(wt))}.reduceByKey(_+_)

    //Numerator for Prediction

    val numerator_predict = user_join_map1.map{ case((user,item_test),(wt,rating_train))=>

      ((user,item_test),wt*rating_train)}.reduceByKey(_+_)

    //prediction

    val prediction = numerator_predict.join(den_predict).map{
      case ((user,item_test),(num:Double,denom:Double)) =>
        if(denom == 0)
          ((userMap_rev.get(user).get,itemMap_rev.get(item_test).get),avg_lookup.getOrElse(item_test,0.0))
        else
          ((userMap_rev.get(user).get,itemMap_rev.get(item_test).get),avg_lookup.getOrElse(item_test,0.0)+ (num/denom))
    }


    //Cold start

    val cold = test_format_gen.map{case (item,user)=>((user,item),1)}
    val cold_pred = cold.subtractByKey(prediction).map{
      case ((user,item),cnt)=>
        if(avg_lookup.contains(itemMap.get(item).get)){
          ((user,item),avg_lookup.get(itemMap.get(item).get).get)
        }
        else {
          ((user, item), average_total)
        }

    }
    val total_pred = cold_pred ++ prediction
    val compare = test_rdd.map{ case Array(user:String,item:String,ratings:String)=>((user,item),ratings.toDouble)}.join(total_pred)

    //RMSE

    val MSE = compare.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
    val RMSE = Math.sqrt(MSE)

    val one = sc.accumulator(0)
    val two = sc.accumulator(0)
    val three = sc.accumulator(0)
    val four = sc.accumulator(0)
    val five = sc.accumulator(0)
    compare.foreach(error => {
      val err = Math.abs(error._2._1 - error._2._2)
      if (err < 1 && err >= 0)
        one += 1
      else if (err < 2 && err >= 1)
        two += 1
      else if (err < 3 && err >= 2)
        three += 1
      else if (err < 4 && err >= 3)
        four += 1
       else
        five += 1
    })
//val out = new PrintWriter("src/main/resources/result_dhiraj.txt")
    val out = new PrintWriter("Dhiraj_Ramnani_ItemBasedCF.txt")
    val result = total_pred.sortByKey().collect().toList
    for (pred <- result) {
      val line = pred._1._1 + "," + pred._1._2 + "," + pred._2 + "\n"
      out.write(line)
    }
    out.close()

    println(">= 0 and < 1: " + one)
    println(">= 1 and < 2: " + two)
    println(">= 2 and < 3: " + three)
    println(">= 3 and < 4: " + four)
    println(">= 4: " + five)
    println("RMSE: " + RMSE)
    val end_time = System.nanoTime()
    val time = (end_time - start_time) / 1000000000
    println("Time: " + time + " sec")

  }

}
