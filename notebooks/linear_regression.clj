;; # Linear Regression with Gradient Descent
;; In this notebook we train a linear regression model from scratch on 
;; synthetic data.  To make things slightly more interesting, we use 
;; gradient descent rather than the closed form solution.
;; ## Setup

(ns linear_regression
  (:require [nextjournal.clerk :as clerk] 
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [axpby! axpy mv xpy entry ax dot dim trans sum col mrows]]
             [random :refer [rand-normal!]]
             [real :refer [entry!]]]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.dataset :as ds]))

;; ### Utility Functions
;; First let's define a couple of quality of life functions

(defn ->vec 
  "For converting Neanderthal vectors into Clojure vectors"
  [x] 
  (into [] x))

(defn mean
  "Take the mean of a Neanderthal vector"
  [x]
  (* (/ 1 (dim x)) (sum x)))

(defn ones
  "Create a vector or matrix of ones"
  ([n]
  (entry! (dv n) 1))
  ([n m]
   (entry! (dge n m) 1)))

;; ### Regression Functions
;; Now let's get in to the real meat of the notebook: the functions that
;; define and train the regression models.  We'll introduce the model itself,
;; the R-squared goodness of fit measure, and a bare-bones gradient decscent algorithm.

(defn linear_regression
  "A typical linear equation"
  [params X]
  (axpy (get params :bias) (ones (mrows X))
        (mv X (get params :weights))))

(defn r2
  "The R-squared of a linear regression"
  [Yhat Y]
  (- 1 (/ (dot (axpy -1 Y Yhat)
               (axpy -1 Y Yhat))
          (dot (axpy (* -1 (mean Y)) (ones (dim Y)) Y)
               (axpy (* -1 (mean Y)) (ones (dim Y)) Y)))))

(defn step
  "Perform one step of gradient descent"
  [params X y learning_rate]
  {:weights (axpy (* -1 learning_rate)
                  (mv (trans X)
                      (axpy -1 y (linear_regression params X)))
                  (get params :weights))
   :bias (+ (get params :bias)
            (* (* -1 learning_rate)
               (sum (axpy -1 y (linear_regression params X)))))})

(defn train
  "Train a linear regression model with gradient descent"
  ([params X y learning_rate epochs count]
   (if (= epochs count)
     params 
     (recur (step params X y learning_rate) X y learning_rate epochs (inc count))))
   ([params X y learning_rate epochs]
    (train params X y learning_rate epochs 0)))

;; ## Univariate Regression
;; We'll start with the single variable case so that we can easily
;; visualize what's going on, but the goal is that our code generalizes
;; seamlessly to the multivariate setting.

;; ### Generate Random Data
;; First we generate some synthetic data.  We'll use the line y = 2x as 
;; our baseline and add some random noise.

(def x (rand-normal! (dv 100)))
(def y (axpby! 2 x 1 (rand-normal! (dv 100))))

;; And to cap this section off, let's plot the data we created.s

(clerk/plotly {:data [{:x (->vec x) 
                       :y (->vec y) 
                       :type "scatter" 
                       :mode "markers"}]})

;; ### Initialize Random Parameters
;; Since we're using gradient descent, let's start our model off by
;; initializing some random parameters.  In the 1D case, both the
;; weight and the bias are scalar, but since we want to easily generalize
;; to multivariate regression, let's make the weight a 1D vector.

(def params_init 
  {:weights (rand-normal! (dv 1)) 
   :bias (entry (rand-normal! (dv 1)) 0)})

;; Just for fun, let's create a model with these random weights.
(def yhat_rdm (linear_regression params_init (dge 100 1 x)))

;; How does this model do?  Unless we're very lucky, this R-squared should
;; be quite far from 1.
(r2 yhat_rdm y)

;; Let's see just how bad this model is:
(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat_rdm)
                       :name "regression"}]})

;; ### Train the Model
;; Now let's train the model!  We'll use a learning rate of 0.0001 and
;; train for 1000 epochs.  Both of these numbers were arbitrarily chosen,
;; I'm sure we could have made significantly more efficient choices!  We 
;; will hopefully end up with a weight close to 2 and a bias close to 0,
;; but it won't be exact because of the noise.

(def params_optimal (train params_init (dge 100 1 x) y 0.0001 1000))

;; Awesome, let's create a model with these parameters!

(def yhat (linear_regression params_optimal (dge 100 1 x)))

;; The R-squared of this model will be much closer to 1 than that of the random model.

(r2 yhat y)

;; Finally, let's admire our work!

(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat)
                       :name "regression"}]})

;; ## Multivariate Regression
;; We've seen that our code works for the single-variable case, now let's
;; see if it can also handle multiple variables!

;; ### Generate Random Data
;; Once again we'll just use synthetic data.  We'll add some noise to
;; a set of sample points from the plane z = 3x + 5y.
(def xy (rand-normal! (dge 100 2)))
(def xx (col xy 0))
(def yy (col xy 1))
(def zz (xpy (xpy (ax 3 xx) (ax 5 yy)) (rand-normal! (dv 100))))

;; Just for fun, let's throw this in to a table!

(clerk/table (ds/dataset {:x (->vec xx) :y (->vec yy) :z (->vec zz)}))

;; And let's take a look at our data.

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}]})

;; ### Initialize Random Parameters
;; Just like in the univariate case, we'll start with a random model and
;; use gradient descent to train it.  So let's initialize the parameters
;; randomly.

(def params_init_3d
  {:weights (rand-normal! (dv 2))
   :bias (entry (rand-normal! (dv 1)) 0)})

;; Computing the model...

(def zzhat_rdm (linear_regression params_init_3d xy))

;;... should yield a poor R-squared.
(r2 zzhat_rdm zz)

;; Let's take a look at how poorly this model fits the data.

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}
                      {:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zzhat_rdm)
                       :type "mesh3d"}]})

;; ## Train the Model

;; Now let's whip those parameters into shape!  We'll go with the same
;; hyperparameters as we did earlier.  Hopefully this yields a bias close
;; to 0 and weights close to [3, 5].

(def params_optimal_3d (train params_init_3d xy zz 0.0001 1000))

;; Building our optimal model...

(def zzhat (linear_regression params_optimal_3d xy))

;; ... yields a great R-squared!

(r2 zzhat zz)

;; And to cap this notebook off, let's take a look at our beautiful 3D model!

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}
                      {:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zzhat)
                       :type "mesh3d"}]})

;; ## Using SciCloj

;; We can also use SciCloj.  SciCloj prefers dataset structures as inputs.  Preview:

(clerk/table (ds/dataset {:x (->vec x) :y (->vec y)}))

(def pipe-fn
  (ml/pipeline
   (mm/set-inference-target :y)
   {:metamorph/id :model}
   (mm/model {:model-type :smile.regression/ordinary-least-square})))

(def trained-model
  (pipe-fn {:metamorph/data (ds/dataset {:x (->vec x) :y (->vec y)})
            :metamorph/mode :fit}))

(ml/explain (:model trained-model))