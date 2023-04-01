;;# Basic Neural Networks

;; In this notebook we train a feedforward neural network classifier to learn the donut problem, which 
;; is a problem that is inherently nonlinear and therefore can't be learned by a classical logistic regression 
;; model without feature engineering.

(ns neural_nets
(:require [nextjournal.clerk :as clerk]
          [uncomplicate.neanderthal
           [core :refer [xpy mm mrows dim col ax sum subvector ncols trans mv entry]]
           [native :refer [dv dge]]
           [vect-math :refer [sigmoid relu mul log abs round cos sin fmax copy-sign]]
           [random :refer [rand-normal!]]
           [math :refer [sqrt]]]))

;; ### Utility Functions

;; First let's define some quality of life functions.

(defn ->vec 
  "For converting Neanderthal vectors in to Clojure vectors"
  [x]
  (into [] x))

(defn ones
  "Create a vector or matrix of ones"
  ([n]
   (dv (repeat n 1)))
  ([n m]
  (dge n m (repeat (* n m) 1))))

(defn zeros
  "Create a vector or matrix of zeros"
  ([n]
   (dv (repeat n 0)))
  ([n m]
   (dge n m (repeat (* n m) 0))))

;; ### Neural Network Output

;; The next thing we'll need are some functions for computing and visualizing the output of a neural network.
;; Here we'll use the sigmoid as our activation functions, but the code can be easily adapted to other choices.

(defn forward_pass
  "A forward pass through a feedforward neural network binary classifier"
  [params Zs]
  (let [n (- (count Zs) 1)
        Z (nth Zs n) 
        W (nth (get params :weights) n)
        b (nth (get params :biases) n)
        Znew (xpy (mm Z W) (dge (mrows Z) (dim b) (mapv #(repeat (mrows Z) %) (->vec b))))] 
  (if (= (count Zs) (count (get params :weights)))
    (conj Zs (sigmoid Znew))
    (recur params (conj Zs (sigmoid Znew))))))

(defn neural_net
  "The output of a feedforward neural network binary classifier"
  [params X]
  (col (last (forward_pass params [X])) 0))

(defn decision_boundary
  "Approximates the decision boundary of a neural network"
  [params]
  (let [mesh_grid (dge 6561 2 [(flatten (repeat 81 (->vec (range -4 4.1 0.1))))
                               (flatten (map #(repeat 81 %) (->vec (range -4 4.1 0.1))))])]
    (-> params
        (neural_net mesh_grid)
        (round)
        (->vec)
        (#(let [z %] {:x (->vec (col mesh_grid 0))
                      :y (->vec (col mesh_grid 1))
                      :z z})))))

;; ### Performance Metrics

;; Next we need ways to monitor the performance of our model as it learns.

(defn cross_entropy
  "Binary cross-entropy loss function"
  [t p]
  (/ (sum (xpy (mul t (log p)) 
            (mul (xpy (ones (dim t)) (ax -1 t))
                 (log (xpy (ones (dim p)) (ax -1 p))))))
  (* -1 (dim t))))

(defn accuracy
  "The accuracy of a prediction"
  [t p]
  (- 1 (/ (sum (abs (xpy t (ax -1 (round p))))) (dim t))))

;; ### Gradient Descent

;; And of course, we need some functions to actually train our networks!

(defn dsigmoid
  "Vectorized derivative of the sigmoid function"
  [Z]
  (mul Z (xpy (ones (mrows Z) (ncols Z)) (ax -1 Z))))

(defn drelu
  "Vectorized derivative of the relu function"
  [Z]
  (fmax (zeros (mrows Z) (ncols Z)) (copy-sign (ones (mrows Z) (ncols Z)) Z)))

(defn backward_pass
  "A backward pass through a feedforward neural network binary classifier"
  [params Zs deltas derivatives]
  (let [n (- (count (get params :weights)) (inc (count (get derivatives :weights))))]
    (if (< n 0)
      derivatives
      (let [W (nth (get params :weights) n)
            Z (nth Zs n)
            dZ (dsigmoid Z)
            delta (first deltas)
            delta_new (mul (mm delta (trans W)) dZ)]
        (recur params 
               Zs 
               (conj deltas delta_new) 
               {:weights (conj (get derivatives :weights) (mm (trans Z) delta))
                :biases (conj (get derivatives :biases) (mv (trans delta) (ones (mrows delta))))})))))

(defn step 
  "Perform one step of gradient descent"
  [params X t learning_rate]
  (let [Zs (forward_pass params [X])
        p (col (last Zs) 0)
        derivatives (backward_pass params
                                   Zs
                                   (list (dge (dim t) 1 (xpy t (ax -1 p))))
                                   {:weights (list) :biases (list)})]
    {:params {:weights (map #(xpy %1 (ax learning_rate %2)) (get params :weights) (get derivatives :weights))
              :biases (map #(xpy %1 (ax learning_rate %2)) (get params :biases) (get derivatives :biases))}
     :loss (cross_entropy t p)
     :accuracy (accuracy t p)}))

(defn train
  "Train the neural network!"
  ([params X t learning_rate epochs history n]
   (let [update (step params X t learning_rate)] 
     (if (= n epochs) 
       {:params params :history history} 
         (recur (get update :params) 
                X 
                t 
                learning_rate 
                epochs
                {:loss (conj (get history :loss) (get update :loss)) 
                 :accuracy (conj (get history :accuracy) (get update :accuracy))} 
                (inc n)))))
  ([params X t learning_rate epochs]
   (train params X t learning_rate epochs {:loss [] :accuracy []} 0)))

;; ### Generating the Data

;; Now we need some data to work with!  As mentioned above, we will attempt to learn the donut problem,
;; which consists of two classes of data points that belong to circles of different radii.  Clearly no
;; straight line does a good job of bisecting the classes, so classical logistic regression does not learn
;; this data very well.

(def theta (ax Math/PI (rand-normal! (dv 200))))

(def xy (dge 200 2 (concat (->vec (cos (subvector theta 0 100)))
                           (->vec (ax 3 (cos (subvector theta 100 100))))
                           (->vec (xpy (sin (subvector theta 0 100)) (ax 0.25 (rand-normal! (dv 100)))))
                           (->vec (xpy (ax 3 (sin (subvector theta 100 100))) (ax 0.25 (rand-normal! (dv 100))))))))

(def labels (dv (concat (repeat 100 0) (repeat 100 1))))

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}]})

;; ### Initializing Random Weights

;; The next step is to randomize some parameters to initialize a model.  We'll use one hidden layer with 8 hidden units, which is 
;; the configuration that was chosen after eshaustive testing... just kidding, it was an arbitrary choice, but we had to have
;; at least one hidden layer!

(def params_init {:weights [(ax (/ 1 (sqrt 2)) (rand-normal! (dge 2 8)))
                            (ax (/ 1 (sqrt 2)) (rand-normal! (dge 8 8)))
                            (ax (/ 1 (sqrt 2)) (rand-normal! (dge 8 1)))]
                  :biases [(ax (/ 1 (sqrt 2)) (rand-normal! (dv 8)))
                            (ax (/ 1 (sqrt 2)) (rand-normal! (dv 8)))
                            (ax (/ 1 (sqrt 2)) (rand-normal! (dv 1)))]})

(def yhat_init (neural_net params_init xy))

;; Now that we have our first model, let's see how it does on the training data.

(cross_entropy labels yhat_init)

(accuracy labels yhat_init)

;; As expected, our accuracy is quite poor... let's visualize this model by plotting it's decision boundary.  In the graph below,
;; the area in blue is where the model predicts a label of 0, while orange represents a label of 1.

(clerk/plotly {:data [{:x (get (decision_boundary params_init) :x)
                       :y (get (decision_boundary params_init) :y)
                       :z (get (decision_boundary params_init) :z)
                       :name "Decision Boundary"
                       :type "contour"
                       :autocontour false
                       :contours {:start -0.5 :end 1.5 :size 1}
                       :opacity 0.35
                       :showscale false}
                      {:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"
                       :marker {:color "#1f77b4"}}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"
                       :marker {:color "#ff7f0e"}}]})

;; ### Training the Network

;; Now let's train our model to see how well it learns the donut!  We'll use a learning rate of 0.001 and train for
;; 1000 iterations ("epochs").

(def optimal (train params_init xy labels 0.001 1000))

;; We can visualize how the training went by plotting the histories of our performance metrics.  In the plot below, the x-axis
;; represents epochs.

(clerk/plotly {:data [{:y (get (get optimal :history) :loss)
                       :name "Loss"}
                      {:y (get (get optimal :history) :accuracy)
                       :name "Accuracy"}]})

(get optimal :params)

;; Finally, let's visualize our optimized model by plotting its decision boundary.

(clerk/plotly {:data [{:x (get (decision_boundary (get optimal :params)) :x)
                       :y (get (decision_boundary (get optimal :params)) :y)
                       :z (get (decision_boundary (get optimal :params)) :z)
                       :name "Decision Boundary"
                       :type "contour"
                       :autocontour false
                       :contours {:start -0.5 :end 1.5 :size 1}
                       :opacity 0.35
                       :showscale false}
                      {:x (->vec (subvector (col xy 0) 0 100))
                       :y (->vec (subvector (col xy 1) 0 100))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"
                       :marker {:color "#1f77b4"}}
                      {:x (->vec (subvector (col xy 0) 100 100))
                       :y (->vec (subvector (col xy 1) 100 100))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"
                       :marker {:color "#ff7f0e"}}]})