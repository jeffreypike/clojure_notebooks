;; # Basical Neural Networks

;; ## Setup

(ns basic_neural_nets
  (:require [nextjournal.clerk :as clerk]
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [subvector mv xpy entry ax dot dim trans sum col cols ncols mrows mm dia rk]]
             [random :refer [rand-normal!]]
             [vect-math :refer [sigmoid relu cos sin log round abs]]]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]))

;; ### Utility Functions
;; First let's define a few quality of life functions.

(defn ->vec
  "For converting Neanderthal vectors into Clojure vectors"
  [x]
  (into [] x))

(defn mat_from_cols
  "Creates a matrix out of a list of columns"
  [columns]
  (dge (dim (nth columns 0))
       (count columns)
       (apply concat (map ->vec columns))))

(defn hadamard_vector
  "Computes the Hadamard product of two vectors"
  [x y]
  (dv (map * (->vec x) (->vec y))))

(defn hadamard
  "Computes the Hadamard product of two matrices"
  [A B]
  (mat_from_cols (map hadamard_vector (cols A) (cols B))))

;; ### Neural Network Functions

(defn forward_pass
  "A forward pass through a basic neural network classifier"
  [params Zs n]
  (if (=  n (- (count (get params :weights)) 1))
    (conj Zs (sigmoid (mm (last Zs) (nth (get params :weights) n))))
    (let [Znew (sigmoid (mm (last Zs) (nth (get params :weights) n)))]
      (recur params (conj Zs Znew) (inc n)))))

(defn neural_net
  "The output of a basic neural network classifier"
  [params X]
  (last (forward_pass params [X] 0)))

(defn backward_pass
  "A backward pass through a basic neural network classifier"
  [params Zs deltas n]
  (let [W (nth (get params :weights) n) Z (nth Zs n)]
    (if (empty? Zs)
      deltas
      (let [delta_new (hadamard
                       (mm (first deltas)
                           (trans W))
                       (hadamard Z
                                 (xpy (dge (mrows Z)
                                           (ncols Z)
                                           (repeat (* (mrows Z) (ncols Z)) 1))
                                      (ax -1 Z))))]
        (recur params Zs (conj deltas delta_new) (- n 1))))))

(defn cross_entropy
  "Binary cross-entropy loss function"
  [p t]
  (/ (sum (xpy (hadamard_vector t (log p))
               (hadamard_vector (xpy (dv (repeat (dim t) 1)) (ax -1 t))
                                (log (xpy (dv (repeat (dim t) 1)) (ax -1 p)))))))
  (* -1 (dim t)))

(defn accuracy
  "The accuracy of a classifier"
  [p t]
  (/ (sum (abs (xpy (round p) (ax -1 t)))) (dim t)))

(defn step
  "Perform one step of gradient descent"
  [params X t learning_rate]
  (let [Zs (forward_pass params [X] 0)]
    {:params {:weights (map (fn [x y] (xpy x (ax (* -1 learning_rate) y)))
                            (get params :weights)
                            (backward_pass params
                                           Zs
                                           (list (xpy t (ax -1 (last Zs))))
                                           (- (count (get params :weights)) -1)))}
     :loss (cross_entropy (last Zs) t)}))

(defn train
  "Train a feedforward neural network with gradient descent"
  ([params X t learning_rate epochs history n]
   (let [update (step params X t learning_rate)]
     (if (= n epochs)
       {:params params :history history}
       (recur (get update :params)
              X
              t
              learning_rate
              epochs
              (conj history (get update :loss))
              (inc n)))))
  ([params X t learning_rate epochs]
   (train params X t learning_rate epochs [] 0)))

(defn estimate_decision_boundary
  "Approximates the decision boundary of a neural network"
  [params]
  (let [xy (dge 5000 2 (ax 3 (rand-normal! (dv 10000))))]
    (-> params
        (neural_net xy)
        (->vec)
        #(tc/dataset {:x (col xy 0)
                      :y (col xy 1)
                      :output %})
        (tc/select-rows (comp #(> % 0.48) :output))
        (tc/select-rows (comp #(< % 0.52) :output))
        (tc/order-by :x)
        #(let [ds %] {:x (->vec (tc/column ds 0))
                      :y (->vec (tc/column ds 1))}))))

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

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 2"
                       :type "scatter"
                       :mode "markers"}
                      {:x (get (estimate_decision_boundary params_init) :x)
                       :y (get (estimate_decision_boundary params_init) :y)
                       :name "Decision Boundary"}]})

;; ### Train the Network

(def optimal (train params_init xy labels 0.0001 1000))

(clerk/plotly {:data [{:y (get optimal :history)}]})

(get optimal :params)

(def yhat (neural_net (get optimal :params) xy))

(cross_entropy yhat labels)

(accuracy yhat labels)

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 2"
                       :type "scatter"
                       :mode "markers"}
                      {:x (get (estimate_decision_boundary (get optimal :params)) :x)
                       :y (get (estimate_decision_boundary (get optimal :params)) :y)
                       :name "Decision Boundary"}]})

