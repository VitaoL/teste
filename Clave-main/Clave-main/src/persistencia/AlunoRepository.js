const Database = require('./database');
const Aluno = require('../dominio/Aluno');

class AlunoRepository {
    async salvar(alunoOuNome, email = null, senhaHash = null) {
        const pool = Database.getPool();
        const aluno = alunoOuNome instanceof Aluno ? alunoOuNome : new Aluno(null, alunoOuNome, email);

        const sql = 'INSERT INTO aluno (nome, email, senha_hash) VALUES ($1, $2, $3) RETURNING id';
        const params = [aluno.obterNome(), aluno.obterEmail(), senhaHash];

        const { rows } = await pool.query(sql, params);
        return rows[0].id;
    }

    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, telefone FROM aluno WHERE id = $1';
        const { rows } = await pool.query(sql, [id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Aluno(row.id, row.nome, row.email, row.telefone);
    }

    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, telefone, senha_hash FROM aluno WHERE email = $1';
        const { rows } = await pool.query(sql, [email]);
        if (rows.length === 0) {
            return null;
        }
        const row = rows[0];
        const aluno = new Aluno(row.id, row.nome, row.email, row.telefone);
        aluno.senha_hash = row.senha_hash;
        return aluno;
    }

    async atualizarContato(id, nome, telefone) {
        const pool = Database.getPool();
        const sql = 'UPDATE aluno SET nome = $1, telefone = $2 WHERE id = $3 RETURNING id, nome, email, telefone';
        const params = [nome, telefone, id];
        const { rows } = await pool.query(sql, params);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Aluno(row.id, row.nome, row.email, row.telefone);
    }
}

module.exports = AlunoRepository;
