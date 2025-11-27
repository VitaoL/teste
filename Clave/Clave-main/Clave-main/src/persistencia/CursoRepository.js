const Database = require('./database');

class CursoRepository {
    async salvar(curso) {
        const pool = Database.getPool();
        const sql = `INSERT INTO curso (professor_id, titulo, descricao, preco, video_url, thumbnail_url) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`;
        const values = [curso.professorId, curso.titulo, curso.descricao, curso.preco, curso.videoUrl, curso.thumbnailUrl || 'https://via.placeholder.com/300x200'];
        const res = await pool.query(sql, values);
        return res.rows[0].id;
    }

    async buscarPorProfessor(professorId) {
        const pool = Database.getPool();
        const sql = 'SELECT * FROM curso WHERE professor_id = $1 ORDER BY criado_em DESC';
        const { rows } = await pool.query(sql, [professorId]);
        return rows;
    }

    // --- ESTE É O MÉTODO IMPORTANTE PARA A HOME ---
    async buscarDestaques() {
        const pool = Database.getPool();
        // Busca dados do curso E o nome/foto do professor dono do curso
        const sql = `
            SELECT c.*, p.nome as professor_nome, p.foto_url as professor_foto 
            FROM curso c
            JOIN professor p ON c.professor_id = p.id
            ORDER BY c.criado_em DESC
        `;
        const { rows } = await pool.query(sql);
        return rows;
    }
}

module.exports = CursoRepository;