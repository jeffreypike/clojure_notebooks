;; # Basical Neural Networks

;; ## Setup

(ns basic_neural_nets
  (:require [nextjournal.clerk :as clerk] 
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [subvector mv xpy entry ax dot dim trans sum col mrows mm dia rk]]
             [random :refer [rand-normal!]]
             [vect-math :refer [sigmoid relu cos sin log round abs]]]
            [tablecloth.api :as tc]
            [clojure.test :as t]
            [tech.v3.dataset :as ds]))

;; ### Utility Functions
;; First let's define a quality of life function.

(defn ->vec
  "For converting Neanderthal vectors into Clojure vectors"
  [x]
  (into [] x))

(defn neural_net
  "The output of a neural network classifier"
  ([params Z n]
   (if (=  n (- (count (get params :weights)) 1))
     (col (sigmoid (mm Z (nth (get params :weights) n))) 0)
     (recur params (relu (mm Z (nth (get params :weights) n))) (inc n))))
  ([params X]
   (neural_net params X 0)))

(defn cross_entropy
  "Binary cross-entropy loss function"
  [p t]
  (/ (sum (xpy (dia (rk t (log p))) 
            (dia (rk (xpy (dv (repeat (dim t) 1))(ax -1 t)) 
                     (log (xpy (dv (repeat (dim t) 1)) (ax -1 p)))))))
     (* -1 (dim t))))

(defn accuracy
  "The accuracy of a classifier"
  [p t]
  (/ (sum (abs (xpy (round p) (ax -1 t)))) (dim t)))

;; ### Generating the Data

(def theta (ax Math/PI (rand-normal! (dv 200))))

(def xy (dge 200 2 (concat (->vec (cos (subvector theta 0 100))) 
                           (->vec (ax 3 (cos (subvector theta 100 100)))) 
                           (->vec (xpy (sin (subvector theta 0 100)) (ax 0.25 (rand-normal! (dv 100))))) 
                           (->vec (xpy (ax 3 (sin (subvector theta 100 100))) (ax 0.25 (rand-normal! (dv 100))))))))

(def labels (dv (concat (repeat 100 0) (repeat 100 1))))

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 2"
                       :type "scatter"
                       :mode "markers"}]})

;; ### Initialize Random Weights

(def params_init {:weights [(rand-normal! (dge 2 4)) (rand-normal! (dge 4 1))]})

(def yhat_rdm (neural_net params_init xy))

(cross_entropy yhat_rdm labels)

(accuracy yhat_rdm labels)

(def span (dge 5000 2 (ax 3 (rand-normal! (dv 10000)))))

(def ds (tc/dataset {:x (->vec (col span 0)) 
                     :y (->vec (col span 1)) 
                     :output (->vec (neural_net params_init span))}))

(def boundary (tc/order-by (tc/select-rows (tc/select-rows ds (comp #(> % 0.48) :output)) (comp #(< % 0.52) :output)) :x))

(clerk/table boundary)

(clerk/plotly {:data [{:x (->vec (tc/column boundary :x))
                       :y (->vec (tc/column boundary :y))}]})