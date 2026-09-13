package com.calvin.posstream;

import net.minecraft.core.BlockPos;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.context.UseOnContext;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StructureScannerItem extends Item {
	/** Override with POSSTREAM_HOST to aim a dev run somewhere else, e.g. 127.0.0.1. */
	static final String HOST = System.getenv().getOrDefault("POSSTREAM_HOST", "100.66.148.86");
	static final int PORT = 5005;
	private static final int CONNECT_TIMEOUT_MS = 1500;

	/** Horizontal half-width of the scanned box, centred on the world origin. */
	private static final int SCAN_RADIUS = 64;
	/** First layer above the grey concrete floor. */
	private static final int MIN_Y = 1;
	private static final int MAX_Y = 64;

	/** Ring of sparks thrown up on a right click, so a scan is visible from outside the chat line. */
	private static final int PARTICLE_COUNT = 48;
	private static final double PARTICLE_RADIUS = 1.6;

	private static final Logger LOGGER = LoggerFactory.getLogger("posstream");

	public StructureScannerItem(Properties properties) {
		super(properties);
	}

	@Override
	public InteractionResult use(Level level, Player player, InteractionHand hand) {
		if (level.isClientSide()) {
			scanAndSend(level, player);
		}

		return InteractionResult.SUCCESS;
	}

	@Override
	public InteractionResult useOn(UseOnContext context) {
		Level level = context.getLevel();
		Player player = context.getPlayer();

		if (level.isClientSide() && player != null) {
			scanAndSend(level, player);
		}

		return InteractionResult.SUCCESS;
	}

	private void scanAndSend(Level level, Player player) {
		spawnScanParticles(level, player);

		List<String> palette = new ArrayList<>();
		Map<Block, Integer> paletteIndices = new HashMap<>();
		List<int[]> blocks = new ArrayList<>();

		int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE, minZ = Integer.MAX_VALUE;
		int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE, maxZ = Integer.MIN_VALUE;

		BlockPos.MutableBlockPos cursor = new BlockPos.MutableBlockPos();

		for (int x = -SCAN_RADIUS; x <= SCAN_RADIUS; x++) {
			for (int z = -SCAN_RADIUS; z <= SCAN_RADIUS; z++) {
				for (int y = MIN_Y; y <= MAX_Y; y++) {
					BlockState state = level.getBlockState(cursor.set(x, y, z));

					if (state.isAir()) {
						continue;
					}

					Block block = state.getBlock();
					Integer index = paletteIndices.get(block);

					if (index == null) {
						index = palette.size();
						paletteIndices.put(block, index);
						palette.add(BuiltInRegistries.BLOCK.getKey(block).toString());
					}

					blocks.add(new int[]{x, y, z, index});

					minX = Math.min(minX, x);
					maxX = Math.max(maxX, x);
					minY = Math.min(minY, y);
					maxY = Math.max(maxY, y);
					minZ = Math.min(minZ, z);
					maxZ = Math.max(maxZ, z);
				}
			}
		}

		if (blocks.isEmpty()) {
			player.displayClientMessage(Component.literal(
					"Nothing found above y=0 within " + SCAN_RADIUS + " blocks of the origin."), false);
			return;
		}

		String json = toJson(palette, blocks, minX, minY, minZ,
				maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1);

		send(player, json, blocks.size());
	}

	private void spawnScanParticles(Level level, Player player) {
		for (int i = 0; i < PARTICLE_COUNT; i++) {
			double angle = (Math.PI * 2 / PARTICLE_COUNT) * i;
			double x = player.getX() + Math.cos(angle) * PARTICLE_RADIUS;
			double z = player.getZ() + Math.sin(angle) * PARTICLE_RADIUS;

			level.addParticle(ParticleTypes.END_ROD, x, player.getY() + 0.2, z, 0.0, 0.12, 0.0);
		}
	}

	private String toJson(List<String> palette, List<int[]> blocks,
			int originX, int originY, int originZ, int sizeX, int sizeY, int sizeZ) {
		StringBuilder json = new StringBuilder(blocks.size() * 16 + 128);

		json.append("{\"origin\":[").append(originX).append(',').append(originY).append(',')
				.append(originZ).append("],\"size\":[").append(sizeX).append(',').append(sizeY)
				.append(',').append(sizeZ).append("],\"count\":").append(blocks.size())
				.append(",\"palette\":[");

		for (int i = 0; i < palette.size(); i++) {
			if (i > 0) {
				json.append(',');
			}

			json.append('"').append(palette.get(i)).append('"');
		}

		json.append("],\"blocks\":[");

		for (int i = 0; i < blocks.size(); i++) {
			int[] entry = blocks.get(i);

			if (i > 0) {
				json.append(',');
			}

			json.append('[').append(entry[0] - originX).append(',').append(entry[1] - originY)
					.append(',').append(entry[2] - originZ).append(',').append(entry[3]).append(']');
		}

		return json.append("]}").toString();
	}

	private void send(Player player, String json, int count) {
		byte[] data = json.getBytes(StandardCharsets.UTF_8);

		try (Socket socket = new Socket()) {
			socket.connect(new InetSocketAddress(HOST, PORT), CONNECT_TIMEOUT_MS);

			OutputStream out = socket.getOutputStream();
			out.write(data);
			out.flush();
			socket.shutdownOutput();

			player.displayClientMessage(Component.literal(
					"Sent " + count + " blocks (" + data.length + " bytes) to " + HOST), false);
		} catch (IOException e) {
			player.displayClientMessage(Component.literal(
					"Could not reach bot at " + HOST + ":" + PORT + " - " + e.getMessage()), false);
			LOGGER.warn("Structure send failed", e);
		}
	}
}
