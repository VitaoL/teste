require('dotenv').config();
const { Pool } = require('pg');

const {
    DB_USER = 'admin',
    DB_HOST = '127.0.0.1',
    DB_NAME = 'meubanco',
    DB_PASSWORD = 'senha123',
    DB_PORT = '5432'
} = process.env;

const pool = new Pool({
    user: DB_USER,
    host: DB_HOST,
    database: DB_NAME,
    password: String(DB_PASSWORD),
    port: Number(DB_PORT)
});

module.exports = {
    getPool() {
        return pool;
    }
};
