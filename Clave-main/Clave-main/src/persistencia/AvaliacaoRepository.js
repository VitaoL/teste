const Database = require('./database');
const Avaliacao = require('../dominio/Avaliacao');

class AvaliacaoRepository {
    
    // CREATE
    async salvar(avaliacao) {
        const pool = Database.getPool();
        // O banco espera: aula_id, aluno_id, professor_id, nota, comentario
        const sql = 'INSERT INTO avaliacao (aula_id, aluno_id, professor_id, nota, comentario) VALUES ($1, $2, $3, $4, $5) RETURNING id';
        const values = [
            avaliacao.obterAulaId(),
            avaliacao.obterAlunoId(),
            avaliacao.obterProfessorId(),
            avaliacao.obterNota(),
            avaliacao.obterComentario()
        ];
        const resultado = await pool.query(sql, values);
        return resultado.rows[0].id;
    }

    // READ - Por Aula
    async buscarPorAulaId(aulaId) {
        const pool = Database.getPool();
        const sql = 'SELECT * FROM avaliacao WHERE aula_id = $1';
        const { rows } = await pool.query(sql, [aulaId]);
        
        if (rows.length === 0) return null;
        const row = rows[0];
        return new Avaliacao(row.id, row.aula_id, row.aluno_id, row.professor_id, row.nota, row.comentario);
    }

    // READ - Por Professor (Novo: Para mostrar na vitrine)
    async buscarPorProfessorId(professorId) {
        const pool = Database.getPool();
        // Faz JOIN para pegar o nome do aluno que avaliou
        const sql = `
            SELECT av.*, al.nome as aluno_nome 
            FROM avaliacao av
            JOIN aluno al ON av.aluno_id = al.id
            WHERE av.professor_id = $1
            ORDER BY av.criado_em DESC
        `;
        const { rows } = await pool.query(sql, [professorId]);
        
        // Retorna um objeto misto com dados do aluno para o frontend
        return rows.map(row => ({
            id: row.id,
            nota: row.nota,
            comentario: row.comentario,
            aluno_nome: row.aluno_nome,
            data: row.criado_em
        }));
    }
    
    // DELETE
    async deletar(id) {
        const pool = Database.getPool();
        const sql = 'DELETE FROM avaliacao WHERE id = $1';
        await pool.query(sql, [id]);
    }
}

module.exports = AvaliacaoRepository;