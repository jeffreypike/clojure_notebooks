(ns linear_regression
  (:require [tablecloth.api :as tc]
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [axpby! mv xpy entry ax dot dim trans sum]]
             [random :refer [rand-normal!]]]
            [nextjournal.clerk :as clerk]))

;; ### Utility Functions

(defn ->vec 
  "For converting Neanderthal vectors into Clojure vectors"
  [x] 
  (into [] x))

;; ### Regression Functions

(defn linear_regression
  "A typical linear equation"
  [params X]
  (xpy (mv X (get params :weights)) (get params :bias)))

(defn mse
  "Mean squared error loss function"
  [Yhat Y]
  (* (/ 1 (dim Y)) 
     (dot (xpy Yhat (ax -1 Y)) 
          (xpy Yhat (ax -1 Y)))))

(defn step
  "Perform one step of gradient descent"
  [params X y learning_rate]
  {:weights (xpy (get params :weights) 
                 (ax (* -1 learning_rate) 
                     (mv (trans X) 
                         (xpy (linear_regression params X) 
                              (ax -1 y)))))
   :bias (xpy (get params :bias)
              (ax (* (* -1 learning_rate)
                  (sum (xpy (linear_regression params X)
                            (ax -1 y))))
                  (dv (repeat (dim (get params :bias)) 1))))})

(defn train
  "Train a linear regression model with gradient descent"
  ([params X y learning_rate epochs count]
   (if (= epochs count)
     params 
     (recur (step params X y learning_rate) X y learning_rate epochs (inc count))))
   ([params X y learning_rate epochs]
    (train params X y learning_rate epochs 0)))

;; ### Generate Random Data

(def x (rand-normal! (dv 100)))
(def y (axpby! 2 x 1 (rand-normal! (dv 100))))

(clerk/table (tc/dataset {:x (->vec x) :y (->vec y)}))

(clerk/plotly {:data [{:x (->vec x) 
                       :y (->vec y) 
                       :type "scatter" 
                       :mode "markers"}]})

;; ### Randomize Initial Parameters

(def params_init 
  {:weights (rand-normal! (dv 1)) 
   :bias (ax (entry (rand-normal! (dv 1)) 0) (dv (repeat 100 1)))})

(def yhat_rdm (linear_regression params_init (dge 100 1 x)))

(mse yhat_rdm y)

(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat_rdm)
                       :name "regression"}]})

;; ### Train the Model
(step params_init (dge 100 1 x) y 0.01)

(def params_optimal (train params_init (dge 100 1 x) y 0.0001 1000))

(def yhat (linear_regression params_optimal (dge 100 1 x)))

(mse yhat y)

(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat)
                       :name "regression"}]})

(mse yhat_rdm y)

(* (/ 1 (dim y))(dot (xpy yhat_rdm (ax -1 y)) (xpy yhat_rdm (ax -1 y))))

(* (/ 1 (dim y)) (dot (xpy yhat (ax -1 y)) (xpy yhat (ax -1 y))))

(mse yhat y)