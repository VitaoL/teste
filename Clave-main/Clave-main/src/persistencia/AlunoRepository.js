const Database = require('./database');
const Aluno = require('../dominio/Aluno');

class AlunoRepository {
    async salvar(aluno) {
        const pool = Database.getPool();
        const sql = 'INSERT INTO aluno (nome, email) VALUES ($1, $2) RETURNING id';
        const { rows } = await pool.query(sql, [aluno.obterNome(), aluno.obterEmail()]);
        return rows[0].id;
    }

    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email FROM aluno WHERE id = $1';
        const { rows } = await pool.query(sql, [id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Aluno(row.id, row.nome, row.email);
    }

    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, senha_hash FROM aluno WHERE email = $1';
        const { rows } = await pool.query(sql, [email]);
        if (rows.length === 0) {
            return null;
        }
        const row = rows[0];
        const aluno = new Aluno(row.id, row.nome, row.email);
        aluno.senha_hash = row.senha_hash;
        return aluno;
    }
}

module.exports = AlunoRepository;
