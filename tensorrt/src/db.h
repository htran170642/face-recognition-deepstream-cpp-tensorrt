#ifndef DB_H
#define DB_H

#include <iostream>
#include <map>
#include <sqlite3.h>

#include "arcface.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Database {
  public:
    Database(std::string path, int embedding_dim);
    ~Database();
    int insertUser(std::string userId, std::string userName);
    int insertFace(std::string userId, std::string imgPath, float embedding[]);
    int deleteUser(std::string userId);
    int deleteFace(int id);
    std::map<std::string, std::string> getUserDict();
    int getNumEmbeddings();
    int getEmbeddings(ArcFaceR100 &recognizer);
    int exportToJson();

  private:
    sqlite3 *m_db;
    int m_embedding_dim;
};

#endif // DB_H
