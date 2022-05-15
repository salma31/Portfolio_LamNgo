import time


import os
import flask
import json
import random as rd
from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import pymongo as pm

class DBManager:
    def __init__(self, database='example', host="db", user="root", password='password'):
        self.connection = None
        while not self.connection:
            time.sleep(1)
            self.connection = mysql.connector.connect(
                user=user, 
                password=password,
                host=host,
                database=database,
                port = 3306,
                auth_plugin='mysql_native_password'
            )
            print("Retrying connection to SQL Database", flush =True)
        print("SQL Database successfully started", flush =True)
        print(48*"_", flush =True)
        self.cursor = self.connection.cursor()
    
    def populate_db(self):
        users = pd.read_csv('rand.csv', sep=",", header=None, names=["first", "last", "email"])
        users = users.drop_duplicates(subset=['email'])
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS user')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE user (email VARCHAR(255) PRIMARY KEY, firstname VARCHAR(255), lastname  VARCHAR(255))')
        self.cursor.executemany('INSERT INTO user (email,firstname, lastname) VALUES (%s, %s, %s)', [(row['email'], row['first'], row['last']) for index, row in users.iterrows()])

        albums = pd.read_csv('album.csv', sep=",", header=None, names=["title", "date"])
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS album')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE album (albumId INT AUTO_INCREMENT PRIMARY KEY, title VARCHAR(255), releaseYear INT)')
        self.cursor.executemany('INSERT INTO album (title, releaseYear) VALUES (%s, %s)', [(row['title'], rd.randint(1900, 2020)) for index, row in albums.iterrows()])

        artists = pd.read_csv('artist.csv', sep=",", header=None, names=["number", "Name", "Genre"])
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS artist')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE artist (artistId INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), artistGenre VARCHAR(255))')
        self.cursor.executemany('INSERT INTO artist (name, artistGenre) VALUES (%s, %s)', [(row['Name'], row['Genre']) for index, row in artists.iterrows()])  

        songs = pd.read_csv('song.csv', sep=",", header=None, names=["TNr", "Name"])
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS song')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE song (trackNumber INT, songTitle VARCHAR(255), songLengthSeconds INT, albumId INT, FOREIGN KEY (albumId) REFERENCES album(albumId)  ON DELETE CASCADE)')
        self.cursor.execute('ALTER TABLE song ADD CONSTRAINT PK_song PRIMARY KEY (trackNumber,albumId)')
        songEntries = []
        columns = ['Tnr','Name', 'Length', "albumid"]
        for index, row in songs.iterrows():
            songEntries.append([row['TNr'], row['Name'], rd.randint(60, 300), rd.randint(1, len(albums.index))])
        songEntriesdf = pd.DataFrame(songEntries, columns=columns)
        self.cursor.executemany('INSERT INTO song (trackNumber, songTitle, songLengthSeconds, albumId) VALUES (%s, %s, %s, %s)', [(row['Tnr'], row['Name'], row["Length"], row["albumid"]) for index, row in songEntriesdf.iterrows()])

        playlists = pd.read_csv('playlist.csv', sep=",", header=None, names=["Name"])
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS playlist')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE playlist (playListId INT AUTO_INCREMENT PRIMARY KEY, playlistTitle VARCHAR(255), owner VARCHAR(255), FOREIGN KEY (owner) REFERENCES user(email))')
        self.cursor.executemany('INSERT INTO playlist (playlistTitle, owner) VALUES (%s, %s)', [(row['Name'], np.random.choice((users["email"]), 1)[0]) for index, row in playlists.iterrows()])

        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        self.cursor.execute('DROP TABLE IF EXISTS playlistSong')
        self.cursor.execute('SET FOREIGN_KEY_CHECKS = 1')
        self.cursor.execute('CREATE TABLE playlistSong (playListId INT, trackNumber INT, albumId INT)')
        self.cursor.execute('ALTER TABLE playlistSong ADD CONSTRAINT PK_pls PRIMARY KEY (playlistId, trackNumber,albumId)')
        playlistSongs = []
        for index, row in playlists.iterrows():
            rnsongs = songEntriesdf[['Tnr', 'albumid']].drop_duplicates().sample(n=10)
            for index2, row2 in rnsongs.iterrows():
                playlistSongs.append([rd.randint(1, len(playlists.index)), row2["Tnr"], row2["albumid"]])
                #print([rd.randint(1, len(playlists.index)), row2["Tnr"], row2["albumid"]], flush = True)      
        columns = ['playListId', 'trackNumber', 'albumId']       
        #print(playlistSongs.head(), flush = True)        
        playlistSongsdf = pd.DataFrame(playlistSongs, columns=columns)
        print("PLAYLISTSONG: " + str(playlistSongsdf.head()), flush = True)
        playlistSongsdf = playlistSongsdf.drop_duplicates()
        self.cursor.executemany('INSERT INTO playlistSong (playListId, trackNumber, albumId) VALUES (%s, %s, %s)', [(int(row["playListId"]), int(row["trackNumber"]), int(row["albumId"])) for index, row in playlistSongsdf.iterrows()])


    def queryGetAll(self, table):
        self.cursor.execute('SELECT * FROM %s' %table)
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec
        
    def mergeUserPlayList(self):
        self.cursor.execute('SELECT DISTINCT user.email, (user.firstname), (user.lastname) FROM user join playlist on user.email = playlist.owner')
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec
        
    def findPlaylistsForUser(self, user):
        print('SELECT playlist.playListId FROM user join playlist on user.email = playlist.owner WHERE user.email = \'{}\''.format(user), flush = True)
        self.cursor.execute('SELECT playlist.playListId FROM user join playlist on user.email = playlist.owner WHERE user.email = \'{}\''.format(user))
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec
        
    def mergeSongAlbum(self):
        self.cursor.execute('SELECT song.trackNumber, song.songTitle, song.albumId, album.title, album.releaseYear  FROM song join album on song.albumId = album.albumId')
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec    
        
    def getUniquePlaylist(self):
        self.cursor.execute('SELECT DISTINCT playListId FROM playlistSong')
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec       
        
    def getSongsforPl(self, plId):
        print('GETSONGSFORPL: SELECT DISTINCT * FROM playlistSong WHERE playlistSong.playListId = \'{}\''.format(plId), flush = True)
        self.cursor.execute('SELECT DISTINCT * FROM playlistSong WHERE playlistSong.playListId = \'{}\''.format(plId))
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec  

      
        
    def searchSong(self, tracknr, albumid):
        print("searchSong:" + 'SELECT * FROM song where trackNumber =\'' + str(tracknr) + "\' and albumId=\'" + str(albumid) + "\'" , flush = True) 
        self.cursor.execute('SELECT * FROM song where trackNumber=\'' + str(tracknr) + "\' and albumId=\'" + str(albumid) + "\'" )
        rec = []
        for c in self.cursor:
            rec.append(c)
        print("rec:" + str(rec), flush =True)
        return rec      
        
    def searchPlaylist(self, email):
        print("PL:" + 'SELECT * FROM playlist where owner=\'' + str(email) + "\'", flush = True)
        self.cursor.execute('SELECT * FROM playlist where owner=\'' + str(email) + "\'")
        rec = []
        for c in self.cursor:
            rec.append(c)
        print("recPL:" + str(rec), flush =True)
        return rec  
        
    def addSongToPlaylist(self, tracknr, albumid, playlist):
        print("'INSERT INTO playlistSong (playlistId, trackNumber,albumId) VALUES (%s, %s, %s)'", (playlist, tracknr, albumid), flush = True)
        self.cursor.execute('INSERT INTO playlistSong (playlistId, trackNumber,albumId) VALUES (%s, %s, %s)', (playlist, tracknr, albumid))
        linesAffected = self.cursor.rowcount
        self.connection.commit()  
        return linesAffected  
        
    def elaborateUseCase2(self, albumId, user):
        print("Elaborate2: " +'SELECT COUNT(DISTINCT pls.albumId) FROM playlist pl JOIN user ON pl.owner=user.email JOIN playlistSong pls ON pls.playListId = pl.playListId JOIN song ON  (song.trackNumber = pls.trackNumber) AND (song.albumId = pls.albumId) JOIN album ON album.albumId = song.albumId WHERE song.albumId =\'' + albumId+ '\' AND user.email = \'' + user + '\'', flush = True)
        self.cursor.execute('SELECT COUNT(DISTINCT pls.albumId) FROM playlist pl JOIN user ON pl.owner=user.email JOIN playlistSong pls ON pls.playListId = pl.playListId JOIN song ON  (song.trackNumber = pls.trackNumber) AND (song.albumId = pls.albumId) JOIN album ON album.albumId = song.albumId WHERE song.albumId =\'' + albumId+ '\' AND user.email = \'' + user + '\'')  
        result = self.cursor.fetchone()
        return result
        
class MongoDBManager:
    def __init__(self, host="mongo", user="root", password='password'):
        self.client = pm.MongoClient(
            username=user, 
            password=password,
            host=host
        )
        self.mydb = self.client["mydatabase"]
   
    def getDB(self):
        return self.mydb
    
    def mongoQueryGetAll(self, collection):
        mycol = self.mydb[collection]
        cursor = mycol.find({})
        rec = []
        for c in cursor:
            rec.append(c)
        return rec        

server = flask.Flask(__name__)
conn = None
MongoConn = None

@server.route('/elaborateUseCase2')
def elaborateUseCase2():
    args = flask.request.args
    user = args["email"]
    albumId = args["albumid"]
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.elaborateUseCase2(albumId, user)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": rec})




@server.route('/transferToMongo')
def transferToMongo():
    global MongoConn
    if not MongoConn:
        MongoConn = MongoDBManager()
    global conn
    if not conn:
        conn = DBManager()
    mydb = MongoConn.getDB()
    userCol = mydb["users"]
    #userCol = mydb["users"]
    #mydict = { "name": "Peter", "address": "Lowstreet 27" }
    rec = conn.mergeUserPlayList()

    for entry in rec:
        try:
            mydict = {}
            mydict["_id"] = entry[0]
            mydict["first"] = entry[1]
            mydict["last"] = entry[2]
            #mydict["playListid"] = entry[3]
            mydict["playListid"] = conn.findPlaylistsForUser(entry[0])
            userCol.insert_one(mydict)
        except pm.errors.DuplicateKeyError:
            continue


        
    cursor = userCol.find({})
    for document in cursor:
          print(document, flush = True)
    
    songCol = mydb["songs"]
    rec = conn.mergeSongAlbum()
    for entry in rec:
        try:
            mydict = {}
            mydict2 = {}
            mydict["trackNumber"] = entry[0]
            mydict["songTitle"] = entry[1]
            mydict2["albumId"] = entry[2]
            mydict2["title"] = entry[3]
            mydict2["releaseYear"] = entry[4]
            mydict["albums"] = mydict2
            songCol.insert_one(mydict)
        except pm.errors.DuplicateKeyError:
            continue
    #x = mycol.insert_one(mydict)
    #print(x.inserted_id, flush= True)
    cursor = songCol.find({})
    for document in cursor:
          print(document, flush = True)
          
    playlistCol = mydb["playlists"]
    rec = conn.getUniquePlaylist()
    for entry in rec:
        try:
            mydict = {}
            mydict["_id"] = entry[0]
            songList = []
            rec2 = conn.getSongsforPl(entry[0])
            for entry2 in rec2: 
                songList = songList + list(songCol.find({"trackNumber": entry2[1],"albums.albumId": entry2[2]},  {"trackNumber": 0, "albums": 0, "songTitle": 0, "_id": 1}))
            print(songList, flush = True)
            songList2 = []
            for entry3 in songList:
                for v in  entry3.values():
                    songList2.append(v)
            mydict["songs"] = songList2
            playlistCol.insert_one(mydict)
        except pm.errors.DuplicateKeyError:
            continue
        
        
    cursor = playlistCol.find({})
    for document in cursor:
        print(document, flush = True)
    return "Successfully transferred the data to MongoDB"

@server.route('/mongoUsers')
def listMongoUsers():
    global MongoConn
    if not MongoConn:
        MongoConn = MongoDBManager()
    rec = MongoConn.mongoQueryGetAll("users")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": result})

@server.route('/mongoSongs')
def listMongoSongs():
    global MongoConn
    if not MongoConn:
        MongoConn = MongoDBManager()
    rec = MongoConn.mongoQueryGetAll("songs")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return str(result)

@server.route('/mongoPlaylists')
def listMongoPlaylists():
    global MongoConn
    if not MongoConn:
        MongoConn = MongoDBManager()
    rec = MongoConn.mongoQueryGetAll("playlists")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return str(result)

@server.route('/albums')
def listAlbums():
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.queryGetAll("album")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": result})

@server.route('/playlistSongs')
def listplaylistSong():
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.queryGetAll("playlistSong")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": result})


@server.route('/songs')
def listSongs():
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.queryGetAll("song")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": result})

@server.route('/playlists')
def listPlaylists():
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.queryGetAll("playlist")
    result = []
    for c in rec:
        result.append(c)
    #print("results:" + str(result), flush=True)
    return flask.jsonify({"response": result})


@server.route('/users')
def listUsers():
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.queryGetAll("user")
    result = []
    for c in rec:
        result.append(c)
    return flask.jsonify({"response": result})

@server.route('/song')
def searchSong():
    args = flask.request.args
    print("args:" + str(args), flush=True)
    tracknr = args["tracknr"]
    albumid = args["albumid"]
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.searchSong(tracknr, albumid)

    result = []
    for c in rec:
        result.append(c)
    if not result:
        print("Returned empty string", flush= True)
        return "Song not found"
    return flask.jsonify({"response": result})

@server.route('/addSongToPlaylist')
def addSongToPlaylist():
    args = flask.request.args
    print("args:" + str(args), flush=True)
    tracknr = args["tracknr"]
    albumid = args["albumid"]
    playlist = args["playlist"]
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.addSongToPlaylist(tracknr, albumid, playlist)

    if (rec==0):
        return "Could not insert"
    return "Successfully inserted"

@server.route('/userEmail')
def searchPL():
    args = flask.request.args
    print("args:" + str(args), flush=True)
    email = args["email"]
    global conn
    if not conn:
        conn = DBManager()
    rec = conn.searchPlaylist(email)

    result = []
    for c in rec:
        result.append(c)
    if not result:
        print("Returned empty string", flush= True)
        return "This user does not exist or does not have a playlist"
    return flask.jsonify({"response": result})


@server.route('/init')
def init():
    global conn
#    if not conn:
    conn = DBManager()
    conn.populate_db()
    #return flask.jsonify({"response": "Hello from Docker!"})
    return "SQL Database successfully created" 

if __name__ == '__main__':
    #time.sleep(5)
    print("START" , flush =True)
    server.run(debug=True, host='0.0.0.0', port=5000)
    #server.run(debug=True)