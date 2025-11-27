require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
    user: 'admin',
    host: '127.0.0.1',
    database: 'meubanco',
    password: process.env.DB_PASSWORD, // aqui vamos colocar senha123 no .env
    port: 5432,
});

module.exports = pool;
