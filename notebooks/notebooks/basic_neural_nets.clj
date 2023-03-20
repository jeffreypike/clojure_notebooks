;; # Basical Neural Networks

;; ## Setup

(ns basic_neural_nets
  (:require [nextjournal.clerk :as clerk] 
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [axpby! mv xpy entry ax dot dim trans sum col mrows]]
             [random :refer [rand-normal!]]]
            [tablecloth.api :as tc]))
