package com.gale.alchemy.conf

import com.gale.alchemy.forecast.utils.Logs

object Constants extends Serializable with Logs {

  val FM_FACTORS = "/user/odia/mackenzie/product_recos/fm_interactions"

  val TRANSACTIONS = "/user/odia/mackenzie/product_recos/transactions_201604_201609"

  val EMBEDDINGS = "/user/odia/mackenzie/lstm_recos/embeddings"

  val TRAINING = "/user/odia/mackenzie/product_recos/training_data_fm_201604_201608_en_purged"

  val TEST = "/user/odia/mackenzie/product_recos/training_data_fm_201609"

  val RESULTS = "/user/odia/mackenzie/lstm_recos/rslts"

  val DATA_DIR = "/user/odia/mackenzie/lstm_recos/"
}
