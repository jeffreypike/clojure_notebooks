(ns nams
  (:require [nextjournal.clerk :as clerk]
            [tablecloth.api :as tc]
            [uncomplicate.neanderthal
             [core :refer [mv!]]
             [native :refer [dge]]]))

(defn ->vec
  "For converting Neanderthal vectors to Clojure vectors"
  [x]
  (into [x] x))

